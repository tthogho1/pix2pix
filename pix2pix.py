"""pix2pix: image-to-image translation with a conditional GAN (U-Net generator + PatchGAN discriminator).

Standalone script version of Copy_of_pix2pix.ipynb (based on the TensorFlow tutorial,
https://www.tensorflow.org/tutorials/generative/pix2pix).

Usage:
    python pix2pix.py --dataset facades --steps 40000
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
LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_dataset(dataset_name):
    url = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"
    path_to_zip = tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz", origin=url, extract=True
    )
    path_to_zip = pathlib.Path(path_to_zip)
    extraction_dir = f"{dataset_name}_extracted/{dataset_name}"
    return path_to_zip.parent / extraction_dir


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Source images are two 256x256 halves side by side: [real | input]
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def build_datasets(path, batch_size, buffer_size):
    train_dataset = tf.data.Dataset.list_files(str(path / "train/*.jpg"))
    train_dataset = train_dataset.map(
        load_image_train, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)

    try:
        test_dataset = tf.data.Dataset.list_files(str(path / "test/*.jpg"))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(str(path / "val/*.jpg"))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Generator (U-Net)
# ---------------------------------------------------------------------------

def downsample(filters, size, apply_batchnorm=True):
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
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
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
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
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


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(
        tf.ones_like(disc_generated_output), disc_generated_output
    )
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, 6)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=initializer
    )(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def generate_images(model, test_input, tar, save_path=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def make_train_step(generator, discriminator, generator_optimizer,
                     discriminator_optimizer, summary_writer):
    @tf.function
    def train_step(input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator(
                [input_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
            tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)

    return train_step


def fit(train_ds, test_ds, steps, train_step, generator, checkpoint_manager,
        sample_dir=None):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if step % 1000 == 0:
            if step != 0:
                print(f"Time taken for 1000 steps: {time.time() - start:.2f} sec\n")
            start = time.time()

            save_path = (
                os.path.join(sample_dir, f"step_{step:07d}.png")
                if sample_dir
                else None
            )
            generate_images(generator, example_input, example_target, save_path)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step)

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
        "--dataset",
        default="facades",
        choices=["cityscapes", "edges2handbags", "edges2shoes", "facades", "maps", "night2day"],
    )
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to an already-downloaded/extracted dataset directory "
            "(containing train/ and test/ or val/ subfolders). If set, "
            "skips downloading --dataset from the internet."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=400)
    parser.add_argument("--checkpoint-dir", default="./training_checkpoints")
    parser.add_argument("--log-dir", default="logs/")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument(
        "--restore", action="store_true", help="restore from the latest checkpoint before training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path = pathlib.Path(args.data_dir) if args.data_dir else download_dataset(args.dataset)
    train_dataset, test_dataset = build_datasets(
        path, args.batch_size, args.buffer_size
    )

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, args.checkpoint_dir, max_to_keep=3
    )

    if args.restore:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    summary_writer = tf.summary.create_file_writer(
        args.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    os.makedirs(args.sample_dir, exist_ok=True)

    train_step = make_train_step(
        generator, discriminator, generator_optimizer, discriminator_optimizer,
        summary_writer,
    )

    fit(
        train_dataset,
        test_dataset,
        args.steps,
        train_step,
        generator,
        checkpoint_manager,
        args.sample_dir,
    )

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    for i, (inp, tar) in enumerate(test_dataset.take(5)):
        generate_images(
            generator, inp, tar, os.path.join(args.sample_dir, f"final_{i}.png")
        )


if __name__ == "__main__":
    main()
