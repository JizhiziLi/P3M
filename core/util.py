"""
Privacy-Preserving Portrait Matting [ACM MM-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au) and Sihan Ma (sima7436@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/P3M
Paper link : https://dl.acm.org/doi/10.1145/3474085.3475512

"""
import os
import shutil
import cv2
import numpy as np
import torch
from config import *

##########################
### Pure functions
##########################
def extract_pure_name(original_name):
	pure_name, extention = os.path.splitext(original_name)
	return pure_name

def listdir_nohidden(path):
	new_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			new_list.append(f)
	new_list.sort()
	return new_list

def create_folder_if_not_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def refresh_folder(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	else:
		shutil.rmtree(folder_path)
		os.makedirs(folder_path)

def save_test_result(save_dir, predict):
	predict = (predict * 255).astype(np.uint8)
	cv2.imwrite(save_dir, predict)

def generate_composite_img(img, alpha_channel):
	b_channel, g_channel, r_channel = cv2.split(img)
	b_channel = b_channel * alpha_channel
	g_channel = g_channel * alpha_channel
	r_channel = r_channel * alpha_channel
	alpha_channel = (alpha_channel*255).astype(b_channel.dtype)	
	img_BGRA = cv2.merge((r_channel,g_channel,b_channel,alpha_channel))
	return img_BGRA

##########################
### for dataset processing
##########################
def trim_img(img):
	if img.ndim>2:
		img = img[:,:,0]
	return img

def gen_trimap_with_dilate(alpha, kernel_size):	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)
	erode = cv2.erode(fg, kernel, iterations=1)
	trimap = erode *255 + (dilate-erode)*128
	return trimap.astype(np.uint8)

##########################
### Functions for fusion 
##########################
def gen_trimap_from_segmap_e2e(segmap):
	trimap = np.argmax(segmap, axis=1)[0]
	trimap = trimap.astype(np.int64)	
	trimap[trimap==1]=128
	trimap[trimap==2]=255
	return trimap.astype(np.uint8)

def get_masked_local_from_global(global_sigmoid, local_sigmoid):
	values, index = torch.max(global_sigmoid,1)
	index = index[:,None,:,:].float()
	### index <===> [0, 1, 2]
	### bg_mask <===> [1, 0, 0]
	bg_mask = index.clone()
	bg_mask[bg_mask==2]=1
	bg_mask = 1- bg_mask
	### trimap_mask <===> [0, 1, 0]
	trimap_mask = index.clone()
	trimap_mask[trimap_mask==2]=0
	### fg_mask <===> [0, 0, 1]
	fg_mask = index.clone()
	fg_mask[fg_mask==1]=0
	fg_mask[fg_mask==2]=1
	fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
	return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
	weighted_global = np.ones(global_result.shape)
	weighted_global[global_result==255] = 0
	weighted_global[global_result==0] = 0
	fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
	return fusion_result

#######################################
### Function to generate training data
#######################################
def generate_paths_for_dataset(args):
	ORI_PATH = DATASET_PATHS_DICT['P3M10K']['TRAIN']['ORIGINAL_PATH']
	MASK_PATH = DATASET_PATHS_DICT['P3M10K']['TRAIN']['MASK_PATH']
	FG_PATH = DATASET_PATHS_DICT['P3M10K']['TRAIN']['FG_PATH']
	BG_PATH = DATASET_PATHS_DICT['P3M10K']['TRAIN']['BG_PATH']	
	mask_list = listdir_nohidden(MASK_PATH)
	total_number = len(mask_list)
	paths_list = []
	for mask_name in mask_list:
		path_list = []
		ori_path = ORI_PATH+extract_pure_name(mask_name)+'.jpg'
		mask_path = MASK_PATH+mask_name
		fg_path = FG_PATH+mask_name
		bg_path = BG_PATH+extract_pure_name(mask_name)+'.jpg'
		path_list.append(ori_path)	
		path_list.append(mask_path)	
		path_list.append(fg_path)
		path_list.append(bg_path)
		paths_list.append(path_list)
	return paths_list