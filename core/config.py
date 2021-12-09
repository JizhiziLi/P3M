"""
Privacy-Preserving Portrait Matting [ACM MM-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au) and Sihan Ma (sima7436@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/P3M
Paper link : https://dl.acm.org/doi/10.1145/3474085.3475512

"""

########## Root Paths and logging files paths
REPOSITORY_ROOT_PATH = ''
P3M_DATASET_ROOT_PATH = ''
TRAIN_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/train_logs/'
TEST_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/test_logs/'

######### Paths of datasets
DATASET_PATHS_DICT={
'P3M10K':{
	'TRAIN':{
		'ROOT_PATH':P3M_DATASET_ROOT_PATH+'train/',
		'ORIGINAL_PATH':P3M_DATASET_ROOT_PATH+'train/blurred_image/',
		'MASK_PATH':P3M_DATASET_ROOT_PATH+'train/mask/',
		'FG_PATH':P3M_DATASET_ROOT_PATH+'train/fg/',
		'BG_PATH':P3M_DATASET_ROOT_PATH+'train/bg/',
		'SAMPLE_NUMBER':9421
		},
	'VAL500P':{
		'ROOT_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-P/',
		'ORIGINAL_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-P/blurred_image/',
		'MASK_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-P/mask/',
		'TRIMAP_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-P/trimap/',
		'SAMPLE_NUMBER':500
		},
	'VAL500NP':{
		'ROOT_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-NP/',
		'ORIGINAL_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-NP/original_image/',
		'MASK_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-NP/mask/',
		'TRIMAP_PATH':P3M_DATASET_ROOT_PATH+'P3M-500-NP/trimap/',
		'SAMPLE_NUMBER':500
		},

	},
}
########## Parameters for training
CROP_SIZE = [512, 768, 1024]
RESIZE_SIZE = 512

########## Parameters for testing
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
SHORTER_PATH_LIMITATION=1080
SAMPLES_ORIGINAL_PATH = REPOSITORY_ROOT_PATH+'samples/original/'
SAMPLES_RESULT_ALPHA_PATH = REPOSITORY_ROOT_PATH+'samples/result_alpha/'
SAMPLES_RESULT_COLOR_PATH = REPOSITORY_ROOT_PATH+'samples/result_color/'
PRETRAINED_P3M10K_MODEL = REPOSITORY_ROOT_PATH+'models/pretrained/p3mnet_pretrained_on_p3m10k.pth'
PRETRAINED_R34_MP = REPOSITORY_ROOT_PATH+'models/pretrained/r34mp_pretrained_imagenet.pth.tar'