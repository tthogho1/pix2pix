"""CycleGAN: unpaired image-to-image translation (e.g. photo <-> anime).

Unlike pix2pix.py, this needs no paired data: it learns from two independent
folders of images and keeps translations consistent by requiring that a round
trip through both generators returns the original image.

Based on the TensorFlow CycleGAN tutorial,
https://www.tensorflow.org/tutorials/generative/cyclegan

Expected dataset layout (the conventional CycleGAN one):

    <data-dir>/
    |-- trainA/   domain A, e.g. real face photos
    |-- trainB/   domain B, e.g. anime faces
    |-- testA/
    `-- testB/

Usage:
    python cyclegan.py --data-dir datasets/photo2anime --steps 100000
"""

import argparse
import datetime
import os
import pathlib
import time

import tensorflow as tf
from matplotlib import pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.cast(image, tf.float32)


def normalize(image):
    return (image / 127.5) - 1


def random_jitter(image):
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image


def load_image_train(image_file):
    return normalize(random_jitter(load(image_file)))


def load_image_test(image_file):
    image = tf.image.resize(
        load(image_file),
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return normalize(image)


def build_dataset(directory, batch_size, buffer_size, training):
    directory = pathlib.Path(directory)
    files = [str(p) for p in sorted(directory.iterdir()) if p.suffix.lower() in
             (".jpg", ".jpeg", ".png", ".bmp")]
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(
        load_image_train if training else load_image_test,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        dataset = dataset.shuffle(buffer_size).repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def instance_norm():
    # groups=-1 puts every channel in its own group, which is instance norm.
    # CycleGAN uses it rather than batch norm so each image is normalized
    # independently of whatever else shares its batch.
    return tf.keras.layers.GroupNormalization(groups=-1)


def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_norm:
        result.add(instance_norm())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(instance_norm())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


# ---------------------------------------------------------------------------
# Generator (U-Net) and Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (bs, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    # Unlike pix2pix's conditional discriminator, this judges a single image:
    # there is no paired input to condition on.
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")

    down1 = downsample(64, 4, apply_norm=False)(inp)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    norm1 = instance_norm()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=initializer
    )(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def discriminator_loss(real, generated):
    real_loss = loss_object(tf.ones_like(real), real)
    generated_loss = loss_object(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5


def generator_loss(generated):
    return loss_object(tf.ones_like(generated), generated)


def cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def generate_images(model, test_input, save_path=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(12, 6))

    display_list = [test_input[0], prediction[0]]
    title = ["Input Image", "Translated Image"]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def make_train_step(generator_g, generator_f, discriminator_x, discriminator_y,
                    generator_g_optimizer, generator_f_optimizer,
                    discriminator_x_optimizer, discriminator_y_optimizer,
                    summary_writer):
    @tf.function
    def train_step(real_x, real_y, step):
        # persistent=True because four networks are differentiated separately.
        with tf.GradientTape(persistent=True) as tape:
            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # Feeding a generator an image already in its target domain should
            # leave it unchanged; this is what keeps colors from drifting.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)
            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(
                real_y, cycled_y
            )

            total_gen_g_loss = (
                gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            )
            total_gen_f_loss = (
                gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
            )

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_optimizer.apply_gradients(
            zip(
                tape.gradient(total_gen_g_loss, generator_g.trainable_variables),
                generator_g.trainable_variables,
            )
        )
        generator_f_optimizer.apply_gradients(
            zip(
                tape.gradient(total_gen_f_loss, generator_f.trainable_variables),
                generator_f.trainable_variables,
            )
        )
        discriminator_x_optimizer.apply_gradients(
            zip(
                tape.gradient(disc_x_loss, discriminator_x.trainable_variables),
                discriminator_x.trainable_variables,
            )
        )
        discriminator_y_optimizer.apply_gradients(
            zip(
                tape.gradient(disc_y_loss, discriminator_y.trainable_variables),
                discriminator_y.trainable_variables,
            )
        )

        del tape

        with summary_writer.as_default():
            tf.summary.scalar("gen_g_loss", total_gen_g_loss, step=step // 1000)
            tf.summary.scalar("gen_f_loss", total_gen_f_loss, step=step // 1000)
            tf.summary.scalar("cycle_loss", total_cycle_loss, step=step // 1000)
            tf.summary.scalar("disc_x_loss", disc_x_loss, step=step // 1000)
            tf.summary.scalar("disc_y_loss", disc_y_loss, step=step // 1000)

    return train_step


def fit(train_ds, test_ds, steps, train_step, generator_g, checkpoint_manager,
        sample_dir=None):
    example_input = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (real_x, real_y) in train_ds.take(steps).enumerate():
        if step % 1000 == 0:
            if step != 0:
                print(f"Time taken for 1000 steps: {time.time() - start:.2f} sec\n")
            start = time.time()

            save_path = (
                os.path.join(sample_dir, f"step_{step:07d}.png")
                if sample_dir
                else None
            )
            generate_images(generator_g, example_input, save_path)
            print(f"Step: {step // 1000}k")

        train_step(real_x, real_y, step)

        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)

        if (step + 1) % 5000 == 0:
            checkpoint_manager.save()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="dataset root containing trainA/ trainB/ testA/ testB/",
    )
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="./cyclegan_checkpoints")
    parser.add_argument("--log-dir", default="logs/")
    parser.add_argument("--sample-dir", default="cyclegan_samples")
    parser.add_argument(
        "--restore", action="store_true", help="restore from the latest checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = pathlib.Path(args.data_dir)
    train_a = build_dataset(root / "trainA", args.batch_size, args.buffer_size, True)
    train_b = build_dataset(root / "trainB", args.batch_size, args.buffer_size, True)
    test_a = build_dataset(root / "testA", args.batch_size, args.buffer_size, False)

    train_dataset = tf.data.Dataset.zip((train_a, train_b))

    # G translates A -> B (the direction you ultimately want), F translates back.
    generator_g = Generator()
    generator_f = Generator()
    discriminator_x = Discriminator()
    discriminator_y = Discriminator()

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, args.checkpoint_dir, max_to_keep=3
    )

    if args.restore:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    summary_writer = tf.summary.create_file_writer(
        args.log_dir + "cyclegan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    os.makedirs(args.sample_dir, exist_ok=True)

    train_step = make_train_step(
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer,
        summary_writer,
    )

    fit(
        train_dataset,
        test_a,
        args.steps,
        train_step,
        generator_g,
        checkpoint_manager,
        args.sample_dir,
    )

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    for i, inp in enumerate(test_a.take(5)):
        generate_images(
            generator_g, inp, os.path.join(args.sample_dir, f"final_{i}.png")
        )


if __name__ == "__main__":
    main()
