---
title: Face to Anime
emoji: 🎨
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Face to anime — four models side by side

Upload a face photo and compare four ways of turning it into an anime image:

- **pix2pix** — trained on paired data. It reads the photo's Canny edges, so it
  discards the photo's colours and invents new ones.
- **CycleGAN** — trained on unpaired selfie2anime data. It reads the photo directly
  and keeps its colours.
- **AnimeGANv2** (`face_paint_512_v2`) — a pretrained reference to judge the other
  models against.
- **ControlNet + anime diffusion** — Counterfeit-V2.5 steered by the same Canny edges
  pix2pix sees. On CPU this is by far the slowest of the four.

The source code is in the `pix2pix` project.
