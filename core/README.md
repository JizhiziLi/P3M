<h1 align="center">Privacy-Preserving Portrait Matting [ACM MM-21]</h1>

<p align="center">
  <a href="#installation">Installation</a> |
  <!-- <a href="#prepare-datasets">Prepare Datasets</a> | -->
  <!-- <a href="#train-on-am-2k">Train on P3M-10k</a> | -->
  <a href="#pretrained-models">Pretrained Models</a> |
  <!-- <a href="#test-on-am-2k">Test on P3M-10k</a> -->
</p>


## Installation
Requirements:

- Python 3.7.7+ with Numpy and scikit-image
- Pytorch (version>=1.7.1)
- Torchvision (version 0.8.2)

1. Clone this repository

    `git clone https://github.com/JizhiziLi/P3M.git`;

2. Go into the repository

    `cd P3M`;

3. Create conda environment and activate

    `conda create -n gfm python=3.7.7`,

    `conda activate p3m`;

4. Install dependencies, install pytorch and torchvision separately if you need

    `pip install -r requirements.txt`,

    `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`.

Our code has been tested with Python 3.7.7, Pytorch 1.7.1, Torchvision 0.8.2, CUDA 10.2 on Ubuntu 18.04.


## Pretrained Models

Here we provide the model we pretrained on P3M-10k and the backbone we pretrained on ImageNet.

| Model|  Pretrained Backbone on ImageNet | Pretrained P3M-NET on P3M-10k | 
| :----: | :----:| :----: | 
| Google Drive  | <a href="https://drive.google.com/uc?export=download&id=18Pt-klsbkiyonMdGi6dytExQEjzBnHwY">Link</a>| [Link](https://drive.google.com/uc?export=download&id=1Vzbt5NUV-q1KP4M8dDXmzP-3sezYdRSh) |
| <p>Baidu Wangpan<br>(百度网盘)</p> | <p><a href="https://pan.baidu.com/s/1vdMQwtu8lnhtLRPjYFG8rA">Link</a><br>(pw: 2v1t)</p>| <p><a href="https://pan.baidu.com/s/1imTTX6Hx6GfyTJuwhQV3_w">Link</a><br>(pw: jocr)</p>|