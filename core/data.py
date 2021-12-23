"""
Privacy-Preserving Portrait Matting [ACM MM-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au) and Sihan Ma (sima7436@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/P3M
Paper link : https://dl.acm.org/doi/10.1145/3474085.3475512

"""

from config import *
from util import *
import torch
import cv2
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
from torchvision import transforms
from torch.autograd import Variable
from skimage.transform import resize

#########################
## Data transformer
#########################
class MattingTransform(object):
	def __init__(self):
		super(MattingTransform, self).__init__()

	def __call__(self, *argv):
		ori = argv[0]
		h, w, c = ori.shape
		rand_ind = random.randint(0, len(CROP_SIZE) - 1)
		crop_size = CROP_SIZE[rand_ind] if CROP_SIZE[rand_ind]<min(h, w) else 512
		resize_size = RESIZE_SIZE
		### generate crop centered in transition area randomly
		trimap = argv[4]
		trimap_crop = trimap[:h-crop_size, :w-crop_size]
		target = np.where(trimap_crop == 128) if random.random() < 0.5 else np.where(trimap_crop > -100)
		if len(target[0])==0:
			target = np.where(trimap_crop > -100)

		rand_ind = np.random.randint(len(target[0]), size = 1)[0]
		cropx, cropy = target[1][rand_ind], target[0][rand_ind]
		# # flip the samples randomly
		flip_flag=True if random.random()<0.5 else False
		# generate samples (crop, flip, resize)
		argv_transform = []
		for item in argv:
			item = item[cropy:cropy+crop_size, cropx:cropx+crop_size]
			if flip_flag:
				item = cv2.flip(item, 1)
			item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
			argv_transform.append(item)
		return argv_transform

#########################
## Data Loader
#########################
class MattingDataset(torch.utils.data.Dataset):
	def __init__(self, args, transform):
		
		self.samples=[]
		self.transform = transform
		self.logging = args.logging
		self.logging.info('===> Loading training set')
		self.samples += generate_paths_for_dataset(args)
		self.logging.info(f"\t--crop_size: {CROP_SIZE} | resize: {RESIZE_SIZE}")
		self.logging.info("\t--Valid Samples: {}".format(len(self.samples)))

	def __getitem__(self,index):
		# Prepare training sample paths
		ori_path, mask_path, fg_path, bg_path  = self.samples[index]
		ori = np.array(Image.open(ori_path))
		mask = trim_img(np.array(Image.open(mask_path)))
		fg = np.array(Image.open(fg_path))
		bg = np.array(Image.open(bg_path))
		# Generate trimap/dilation/erosion online
		kernel_size = random.randint(15, 30)
		trimap = gen_trimap_with_dilate(mask, kernel_size)
		# Data transformation to generate samples (crop/flip/resize)
		argv = self.transform(ori, mask, fg, bg, trimap)
		argv_transform = []
		for item in argv:
			if item.ndim<3:
				item = torch.from_numpy(item.astype(np.float32)[np.newaxis, :, :])
			else:
				item = torch.from_numpy(item.astype(np.float32)).permute(2, 0, 1)
			argv_transform.append(item)

		[ori, mask, fg, bg, trimap] = argv_transform
		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		ori = ori/255.0
		ori = normalize(ori)
		fg = fg/255.0
		fg = normalize(fg)
		bg = bg/255.0
		bg = normalize(bg)
		return ori, mask, fg, bg, trimap

	def __len__(self):
		return len(self.samples)
