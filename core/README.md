<h1 align="center">Privacy-Preserving Portrait Matting [ACM MM-21]</h1>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#prepare-datasets">Prepare Datasets</a> |
  <a href="#train-on-p3m-10k">Train on P3M-10k</a> |
  <a href="#pretrained-models">Pretrained Models</a> |
  <a href="#test-on-p3m-10k">Test on P3M-10k</a>
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

    `conda create -n p3m python=3.7.7`,

    `conda activate p3m`;

4. Install dependencies, install pytorch and torchvision separately if you need

    `pip install -r requirements.txt`,

    `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`.

Our code has been tested with Python 3.7.7, Pytorch 1.7.1, Torchvision 0.8.2, CUDA 10.2 on Ubuntu 18.04.

## Prepare Datasets

| Dataset | <p>Dataset Link<br>(Google Drive)</p> | <p>Dataset Link<br>(Baidu Wangpan 百度网盘)</p> | Dataset Release Agreement|
| :----:| :----: | :----: | :----: | 
|<strong>P3M-10k</strong>|[Link](https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1)|[Link](https://pan.baidu.com/s/1X9OdopT41lK0pKWyj0qSEA) (pw: fgmc)|[Agreement (MIT License)](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf)| 

1. Download the datasets P3M-10k from the above links and unzip to the folders `P3M_DATASET_ROOT_PATH`, set up the configuratures in the file `core/config.py`. Please make sure that you have checked out and agreed to the agreements.

2. You will need to generate the foregrounds and backgrounds following closed form method as in the paper `Levin, Anat, Dani Lischinski, and Yair Weiss. "A closed-form solution to natural image matting." IEEE transactions on pattern analysis and machine intelligence, 2007` first. Some reference implementations can be referred to [here (python)](https://github.com/MarcoForte/closed-form-matting/blob/master/closed_form_matting/solve_foreground_background.py) and [here (matlab)](http://people.csail.mit.edu/alevin/matting.tar.gz). Inser the results in the folder `DATASET_PATHS_DICT['P3M10K']['TRAIN']['FG_PATH']` and `DATASET_PATHS_DICT['P3M10K']['TRAIN']['BG_PATH']`.

After datasets preparation, the structure of the complete datasets should be like the following. 
```text
P3M-10k
├── train
    ├── blurred_image
    ├── mask (alpha mattes)
├── validation
    ├── P3M-500-P
        ├── blurred_image
        ├── mask
        ├── trimap
    ├── P3M-500-NP
        ├── original_image
        ├── mask
        ├── trimap
```

## Pretrained Models

Here we provide the model we pretrained on P3M-10k and the backbone we pretrained on ImageNet.

| Model|  Pretrained Backbone on ImageNet | Pretrained P3M-NET on P3M-10k | 
| :----: | :----:| :----: | 
| Google Drive  | <a href="https://drive.google.com/uc?export=download&id=18Pt-klsbkiyonMdGi6dytExQEjzBnHwY">Link</a>| [Link](https://drive.google.com/uc?export=download&id=1smX2YQGIpzKbfwDYHAwete00a_YMwoG1) |
| <p>Baidu Wangpan<br>(百度网盘)</p> | <p><a href="https://pan.baidu.com/s/1vdMQwtu8lnhtLRPjYFG8rA">Link</a><br>(pw: 2v1t)</p>| <p><a href="https://pan.baidu.com/s/1zGF3qnnD8qpI-Z5Nz0TDGA">Link</a><br>(pw: 2308)</p>|


## Train on P3M-10k

Here we provide the procedure of training on P3M-10k:

1. Setup the environment following this [section](#installation);

2. Setup required parameters in `core/config.py`;

3.  (1) To train with closed_form foregrounds and backgrounds of AM-2k (the same way as in our paper), run the code:
    
    `chmod +x scripts/train/*`,

    `./scripts/train.sh`;

4. The training logging file will be saved in the file `logs/train_logs/args.logname`;

5. The trained model will be saved in the folder `args.model_save_dir`.


## Test on P3M-10k

1. Create test logging folder `logs/test_logs/`;

2. Download pretrained models as shown in the previous section, unzip to the folder `models/pretrained/`;

3. Download AM-2k dataset in root `P3M_DATASET_ROOT_PATH`;

4. Setup parameters in `scripts/test_dataset.sh`, choose `dataset_choice=P3M_500_NP` or `dataset_choice=P3M_500_P` depends on which validation set you want to use, run the file:

    `chmod +x scripts/test/*`

    `./scripts/test_dataset.sh`

5. The results of the alpha matte will be saved in folder `args.test_result_dir`. The logging files including the evaluation results will be saved in the file `logs/test_logs/args.logname`. Note that there may be some slight differences of the evaluation results with the ones reported in the paper due to some packages versions differences and the testing strategy. Some test results on P3M-10k can be seen [here](https://github.com/JizhiziLi/P3M/tree/master/demo/).