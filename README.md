# Latent Video Diffusion Models (LVDM)

Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths  
*Yingqing He, Tinayu Yang, Yong Zhang, Ying Shan, Qifeng Chen*

## Results
Unconditional video generation results with 1000 frames and spatial resolution of 256 $\times$ 256 produced by LVDM.

<p align="center">
    <img src=assets/sky-long-001.gif />
    <img src=assets/sky-long-002.gif />
    <img src=assets/sky-long-003.gif />
</p>

<p align="center">
    <img src=assets/ucf-long-001.gif />
    <img src=assets/ucf-long-002.gif />
    <img src=assets/ucf-long-003.gif />
</p>


## Abstract
AI-generated content has attracted lots of attention recently, but photo-realistic video synthesis is still challenging. Although many attempts using GANs and autoregressive models have been made in this area, the visual quality and length of generated videos are far from satisfactory. Diffusion models (DMs) are another class of deep generative models and have recently achieved remarkable performance on various image synthesis tasks. However, training image diffusion models usually requires substantial computational resources to achieve a high performance, which makes expanding diffusion models to high-dimensional video synthesis tasks more computationally expensive. To ease this problem while leveraging its advantages, we introduce lightweight video diffusion models that synthesize high-fidelity and arbitrary-long videos from pure noise. Specifically, we propose to perform diffusion and denoising in a low-dimensional 3D latent space, which significantly outperforms previous methods on 3D pixel space when under a limited computational budget. In addition, though trained on tens of frames, our models can generate videos with arbitrary lengths, i.e., thousands of frames, in an autoregressive way. Finally, conditional latent perturbation is further introduced to reduce performance degradation during synthesizing long-duration videos. Extensive experiments on various datasets and generated lengths suggest that our framework is able to sample much more realistic and longer videos than previous approaches, including GAN-based, autoregressive-based, and diffusion-based methods.

## Framework
<p align="center">
    <img src=assets/framework.jpeg />
</p>

We present LVDM, a novel diffusion model (DM)-based framework for video generation. The diffusion and denoising process is performed on the video latent space, which is learned by a *3D autoencoder*. Then an *unconditional DM* is trained on the latent space for generating short video clips. To extend videos to arbitrary lengths, we further propose two frame-conditional models, including a *prediction DM* and an *infilling DM* which can synthesize long-duration videos in autoregressive and hierarchical ways. We utilize noisy conditions at diffusion timestep $s$ to mitigate the condition error induced during the autoregressive sampling process. The frame-conditional DMs are jointly trained with unconditional inputs, where the conditional and unconditional sample frequencies are controlled by their corresponding probabilities, i.e., $p_c$ and $p_u$.

