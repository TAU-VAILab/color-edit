# Color-Edit: Text-guided Multiple Color Editing of Objects

This is the official pytorch implementation of Color-Edit.

[![arXiv](https://img.shields.io/badge/arXiv-2303.12048-b31b1b.svg)](https://arxiv.org/abs/2303.12048)
![Generic badge](https://img.shields.io/badge/conf-ICCV2023-purple.svg)

[[Project Website](https://tau-vailab.github.io/color-edit/)]

> **Not Every Gift Comes in Gold Paper or with a Red Ribbon:<br>
Exploring Color Perception in Text-to-Image Models**<br>
> Shay Shomer-Chai<sup>1</sup>, Wenxuan Peng <sup>2</sup>, Bharath Hariharan<sup>2</sup>, Hadar Averbuch-Elor<sup>2</sup><br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>Cornell University

>**Abstract** <br>
>                Text-to-image generation has recently seen remarkable success, 
                granting users with the ability to create high-quality images through 
                the use of text. However, contemporary methods face challenges in capturing 
                the precise semantics conveyed by complex multi-object prompts. Consequently, 
                many works have sought to mitigate such semantic misalignments, typically via 
                inference-time schemes that modify the attention layers of the denoising networks. 
                However, prior work has mostly utilized coarse metrics, such as the cosine similarity 
                between text and image CLIP embeddings, or human evaluations, which are challenging 
                to conduct on a larger-scale. In this work, we perform a case study on colors---
                a fundamental attribute commonly associated with objects in text prompts, which 
                offer a rich test bed for rigorous evaluation. Our analysis reveals that 
                pretrained models struggle to generate images that faithfully reflect multiple 
                color attributes—far more so than with single-color prompts—and that neither 
                inference-time techniques nor existing editing methods reliably resolve these 
                semantic misalignments.  Accordingly, we introduce a dedicated image editing 
                technique, mitigating the issue of multi-object semantic alignment for prompts 
                containing multiple colors. We demonstrate that our approach significantly 
                boosts performance over a wide range of metrics, considering images generated 
                by various text-to-image diffusion-based techniques.

![Graph](https://tau-vailab.github.io/color-edit/images/teaser_july17.png "Flow:")
</br>

