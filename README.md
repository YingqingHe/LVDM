
<div align="center">

<h2> LVDM: <span style="font-size:12px">Latent Video Diffusion Models for High-Fidelity Long Video Generation </span> </h2> 

  <a href='https://arxiv.org/abs/2211.13221'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://yingqinghe.github.io/LVDM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


<div>
    <a href='https://github.com/YingqingHe' target='_blank'>Yingqing He <sup>1</sup> </a>&emsp;
    <a href='https://tianyu-yang.com/' target='_blank'>Tianyu Yang <sup>2</a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Ying Shan <sup>2</sup></a>&emsp;
    <a href='https://cqf.io/' target='_blank'>Qifeng Chen <sup>1</sup></a>&emsp; </br>
</div>
<br>
<div>
    <sup>1</sup> The Hong Kong University of Science and Technology &emsp; <sup>2</sup> Tencent AI Lab &emsp;
</div>
<br>
<br>

<b>TL;DR: An efficient video diffusion model that can:</b>  
1️⃣ conditionally generate videos based on input text;  
2️⃣ unconditionally generate videos with thousands of frames.

<br>

</div>


## 🍻 Results
### ☝️ Text-to-Video Generation

<table class="center">
  <!-- <td style="text-align:center;" width="50">Input Text</td> -->
  <td style="text-align:center;" width="170">"A corgi is swimming fastly"</td>
  <td style="text-align:center;" width="170">"astronaut riding a horse"</td>
  <td style="text-align:center;" width="170">"A glass bead falling into water with a huge splash. Sunset in the background"</td>
  <td style="text-align:center;" width="170">"A beautiful sunrise on mars. High definition, timelapse, dramaticcolors."</td>
  <td style="text-align:center;" width="170">"A bear dancing and jumping to upbeat music, moving his whole body."</td>
  <td style="text-align:center;" width="170">"An iron man surfing in the sea. cartoon style"</td>
  <tr>
  <td><img src=assets/t2v-001.gif width="170"></td>
  <td><img src=assets/t2v-002.gif width="170"></td>
  <td><img src=assets/t2v-003.gif width="170"></td>
  <td><img src=assets/t2v-007.gif width="170"></td>
  <td><img src=assets/t2v-005.gif width="170"></td>
  <td><img src=assets/t2v-004.gif width="170"></td>
</tr>
</table >

### ✌️ Unconditional Long Video Generation (40 seconds)
<table class="center">
  <td><img src=assets/sky-long-001.gif width="170"></td>
  <td><img src=assets/sky-long-002.gif width="170"></td>
  <td><img src=assets/sky-long-003.gif width="170"></td>
  <td><img src=assets/ucf-long-001.gif width="170"></td>
  <td><img src=assets/ucf-long-002.gif width="170"></td>
  <td><img src=assets/ucf-long-003.gif width="170"></td>
  <tr>
</tr>
</table >

## ⏳ TODO
- [x] Release pretrained text-to-video generation models and inference code
- [x] Release unconditional video generation models
- [x] Release training code
- [ ] Update training and sampling for long video generation
<br>

---
## ⚙️ Setup

### Install Environment via Anaconda
```bash
conda create -n lvdm python=3.8.5
conda activate lvdm
pip install -r requirements.txt
```
### Pretrained Models and Used Datasets

<!-- <div style="text-indent:25px"> -->
<!-- <details><summary></summary> -->
Download via linux commands:
```
mkdir -p models/ae
mkdir -p models/lvdm_short
mkdir -p models/t2v

# sky timelapse
wget -O models/ae/ae_sky.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/ae/ae_sky.ckpt
wget -O models/lvdm_short/short_sky.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/short_sky.ckpt  

# taichi
wget -O models/ae/ae_taichi.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/ae/ae_taichi.ckpt
wget -O models/lvdm_short/short_taichi.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/short_taichi.ckpt

# text2video
wget -O models/t2v/model.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/t2v.ckpt
```
<!-- </details>
</div> -->
<!-- - UCF-101: [dataset](https://www.crcv.ucf.edu/data/UCF101.php) -->
<!-- [samples_short](TBD), [samples_long](TBD) -->

Download manually:
- Sky Timelapse: [VideoAE](https://huggingface.co/Yingqing/LVDM/blob/main/ae/ae_sky.ckpt), [LVDM_short](https://huggingface.co/Yingqing/LVDM/blob/main/lvdm_short/short_sky.ckpt), [LVDM_pred](TBD), [LVDM_interp](TBD), [dataset](https://github.com/weixiong-ur/mdgan)
- Taichi: [VideoAE](https://huggingface.co/Yingqing/LVDM/blob/main/ae/ae_taichi.ckpt), [LVDM_short](https://huggingface.co/Yingqing/LVDM/blob/main/lvdm_short/short_taichi.ckpt), [dataset](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md)
- Text2Video: [model](https://huggingface.co/Yingqing/LVDM/blob/main/lvdm_short/t2v.ckpt)

---
## 💫 Inference 
### Sample Short Videos 
- unconditional generation

```
bash shellscripts/sample_lvdm_short.sh
```
- text to video generation
```
bash shellscripts/sample_lvdm_text2video.sh
```

### Sample Long Videos 
```
bash shellscripts/sample_lvdm_long.sh
```

---
## 💫 Training
<!-- tar -zxvf dataset/sky_timelapse.tar.gz -C /dataset/sky_timelapse -->
### Train video autoencoder
```
bash shellscripts/train_lvdm_videoae.sh 
```
- remember to set `PROJ_ROOT`, `EXPNAME`, `DATADIR`, and `CONFIG`.

### Train unconditional lvdm for short video generation
```
bash shellscripts/train_lvdm_short.sh
```
- remember to set `PROJ_ROOT`, `EXPNAME`, `DATADIR`, `AEPATH` and `CONFIG`.

### Train unconditional lvdm for long video generation
```
# TBD
```

---
## 💫 Evaluation
```
bash shellscripts/eval_lvdm_short.sh
```
- remember to set `DATACONFIG`, `FAKEPATH`, `REALPATH`, and `RESDIR`.
---

## 📃 Abstract
AI-generated content has attracted lots of attention recently, but photo-realistic video synthesis is still challenging. Although many attempts using GANs and autoregressive models have been made in this area, the visual quality and length of generated videos are far from satisfactory. Diffusion models have shown remarkable results recently but require significant computational resources. To address this, we introduce lightweight video diffusion models by leveraging a low-dimensional 3D latent space, significantly outperforming previous pixel-space video diffusion models under a limited computational budget. In addition, we propose hierarchical diffusion in the latent space such that longer videos with more than one thousand frames can be produced. To further overcome the performance degradation issue for long video generation, we propose conditional latent perturbation and unconditional guidance that effectively mitigate the accumulated errors during the extension of video length. Extensive experiments on small domain datasets of different categories suggest that our framework generates more realistic and longer videos than previous strong baselines. We additionally provide an extension to large-scale text-to-video generation to demonstrate the superiority of our work. Our code and models will be made publicly available.
<br>

## 🔮 Pipeline

<p align="center">
    <img src=assets/framework.jpg />
</p>

---
## 😉 Citation

```
@article{he2022lvdm,
      title={Latent Video Diffusion Models for High-Fidelity Long Video Generation}, 
      author={Yingqing He and Tianyu Yang and Yong Zhang and Ying Shan and Qifeng Chen},
      year={2022},
      eprint={2211.13221},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🤗 Acknowledgements
We built our code partially based on [latent diffusion models](https://github.com/CompVis/latent-diffusion) and [TATS](https://github.com/SongweiGe/TATS). Thanks the authors for sharing their awesome codebases! We aslo adopt Xintao Wang's [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling our text-to-video generation results. Thanks for their wonderful work!