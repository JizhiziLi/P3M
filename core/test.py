"""
Privacy-Preserving Portrait Matting [ACM MM-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au) and Sihan Ma (sima7436@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/P3M
Paper link : https://dl.acm.org/doi/10.1145/3474085.3475512

"""
import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torchvision import transforms
import logging
from config import *
from util import *
from evaluate import *
from network.P3mNet import P3mNet


def get_args():
	# Statement of args
	# --cude: use cude or not
	# --deploy: deploy flag
	# --model_path: pretrained model path
	# --test_choice: [HYBRID/RESIZE] test strategy choices
	# --dataset_choice: [P3M_500_P/P3M_500_NP/SAMPLES] test dataset choices
	# --test_result_dir: path to save predict results
	# --logname: name of the logging file
	parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
	parser.add_argument('--cuda', action='store_true', help='use cuda?')
	parser.add_argument('--dataset_choice', type=str, required=True, choices=['P3M_500_P','P3M_500_NP','SAMPLES'], help="which dataset to test")
	parser.add_argument('--model_path', type=str, default='', required=False, help="path of model to use")
	parser.add_argument('--test_choice', type=str, required=True, choices=['RESIZE','HYBRID'], help="which dataset to test")
	parser.add_argument('--test_result_dir', type=str, required=False, default='', help="test result save to")
	parser.add_argument('--logname', type=str, default='test_logs', required=False, help="logging name file for testing")
	args = parser.parse_args()
	return args

def inference_once(args, model, scale_img, scale_trimap=None):
	pred_list = []
	if args.cuda:
		tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
	else:
		tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
	input_t = tensor_img
	input_t = input_t/255.0
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
	input_t = normalize(input_t)
	input_t = input_t.unsqueeze(0)
	pred_global, pred_local, pred_fusion = model(input_t)[:3]
	pred_global = pred_global.data.cpu().numpy()
	pred_global = gen_trimap_from_segmap_e2e(pred_global)
	pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
	pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
	return pred_global, pred_local, pred_fusion

def inference_img(args, model, img):
	h, w, c = img.shape
	new_h = min(MAX_SIZE_H, h - (h % 32))
	new_w = min(MAX_SIZE_W, w - (w % 32))
	if args.test_choice=='HYBRID':
		global_ratio = 1/2
		local_ratio = 1
		resize_h = int(h*global_ratio)
		resize_w = int(w*global_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_coutour_1, pred_retouching_1, pred_fusion_1 = inference_once(args, model, scale_img)
		pred_coutour_1 = resize(pred_coutour_1,(h,w))*255.0
		resize_h = int(h*local_ratio)
		resize_w = int(w*local_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_coutour_2, pred_retouching_2, pred_fusion_2 = inference_once(args, model, scale_img)		
		pred_retouching_2 = resize(pred_retouching_2,(h,w))
		pred_fusion = get_masked_local_from_global_test(pred_coutour_1, pred_retouching_2)
		return pred_fusion
	else:
		resize_h = int(h/2)
		resize_w = int(w/2)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_global, pred_local, pred_fusion = inference_once(args, model, scale_img)
		pred_local = resize(pred_local,(h,w))
		pred_global = resize(pred_global,(h,w))*255.0
		pred_fusion = resize(pred_fusion,(h,w))
		return pred_fusion


def test_p3m10k(args, model):
	############################
	# Some initial setting for paths
	############################
	if args.dataset_choice == 'P3M_500_P':
		val_option = 'VAL500P'
	else:
		val_option = 'VAL500NP'
	ORIGINAL_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['ORIGINAL_PATH']
	MASK_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['MASK_PATH']
	TRIMAP_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['TRIMAP_PATH']
	############################
	# Start testing
	############################
	sad_diffs = 0.
	mse_diffs = 0.
	mad_diffs = 0.
	sad_trimap_diffs = 0.
	mse_trimap_diffs = 0.
	mad_trimap_diffs = 0.
	sad_fg_diffs = 0.
	sad_bg_diffs = 0.
	conn_diffs = 0.
	grad_diffs = 0.
	refresh_folder(args.test_result_dir)
	model.eval()
	img_list = listdir_nohidden(ORIGINAL_PATH)
	total_number = len(img_list)
	args.logging.info("===============================")
	args.logging.info(f'====> Start Testing\n\t--Dataset: {args.dataset_choice}\n\t--Test: {args.test_choice}\n\t--Number: {total_number}')

	for img_name in img_list:
		img_path = ORIGINAL_PATH+img_name
		alpha_path = MASK_PATH+extract_pure_name(img_name)+'.png'
		trimap_path = TRIMAP_PATH+extract_pure_name(img_name)+'.png'
		img = np.array(Image.open(img_path))
		trimap = np.array(Image.open(trimap_path))
		alpha = np.array(Image.open(alpha_path))/255.
		img = img[:,:,:3] if img.ndim>2 else img
		trimap = trimap[:,:,0] if trimap.ndim>2 else trimap
		alpha = alpha[:,:,0] if alpha.ndim>2 else alpha
		with torch.no_grad():
			if args.cuda:
				torch.cuda.empty_cache()
			predict = inference_img(args, model, img)
			sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
			sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)
			sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
			conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
			grad_diff = compute_gradient_whole_image(predict, alpha)
			args.logging.info(f"[{img_list.index(img_name)}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nsad_trimap:{sad_trimap_diff}\nmse_trimap:{mse_trimap_diff}\nmad_trimap:{mad_trimap_diff}\nsad_fg:{sad_fg_diff}\nsad_bg:{sad_bg_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\n-----------")
			sad_diffs += sad_diff
			mse_diffs += mse_diff
			mad_diffs += mad_diff
			mse_trimap_diffs += mse_trimap_diff
			sad_trimap_diffs += sad_trimap_diff
			mad_trimap_diffs += mad_trimap_diff
			sad_fg_diffs += sad_fg_diff
			sad_bg_diffs += sad_bg_diff
			conn_diffs += conn_diff
			grad_diffs += grad_diff
			save_test_result(os.path.join(args.test_result_dir, extract_pure_name(img_name)+'.png'),predict)
				
	args.logging.info("===============================")
	args.logging.info(f"Testing numbers: {total_number}")
	args.logging.info("SAD: {}".format(sad_diffs / total_number))
	args.logging.info("MSE: {}".format(mse_diffs / total_number))
	args.logging.info("MAD: {}".format(mad_diffs / total_number))
	args.logging.info("SAD TRIMAP: {}".format(sad_trimap_diffs / total_number))
	args.logging.info("MSE TRIMAP: {}".format(mse_trimap_diffs / total_number))
	args.logging.info("MAD TRIMAP: {}".format(mad_trimap_diffs / total_number))
	args.logging.info("SAD FG: {}".format(sad_fg_diffs / total_number))
	args.logging.info("SAD BG: {}".format(sad_bg_diffs / total_number))
	args.logging.info("CONN: {}".format(conn_diffs / total_number))
	args.logging.info("GRAD: {}".format(grad_diffs / total_number))

	return int(sad_diffs/total_number)


def test_samples(args, model):
	print(f'=====> Test on samples and save alpha results')
	model.eval()
	img_list = listdir_nohidden(SAMPLES_ORIGINAL_PATH)
	refresh_folder(SAMPLES_RESULT_ALPHA_PATH)
	refresh_folder(SAMPLES_RESULT_COLOR_PATH)
	for img_name in tqdm(img_list):
		img_path = SAMPLES_ORIGINAL_PATH+img_name
		try:
			img = np.array(Image.open(img_path))[:,:,:3]
		except Exception as e:
			print(f'Error: {str(e)} | Name: {img_name}')
		h, w, c = img.shape
		if min(h, w)>SHORTER_PATH_LIMITATION:
		  if h>=w:
			  new_w = SHORTER_PATH_LIMITATION
			  new_h = int(SHORTER_PATH_LIMITATION*h/w)
			  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
		  else:
			  new_h = SHORTER_PATH_LIMITATION
			  new_w = int(SHORTER_PATH_LIMITATION*w/h)
			  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

		with torch.no_grad():
			if args.cuda:
				torch.cuda.empty_cache()
			predict = inference_img(args, model, img)

		composite = generate_composite_img(img, predict)
		cv2.imwrite(os.path.join(SAMPLES_RESULT_COLOR_PATH, extract_pure_name(img_name)+'.png'),composite)
		predict = predict*255.0
		predict = cv2.resize(predict, (w, h), interpolation=cv2.INTER_LINEAR)
		cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))
		

def load_model_and_deploy(args):
	print('*********************************')
	print(f'Loading model: {args.model_path}')
	print(f'Test stategy: {args.test_choice}')
	print(f'Test dataset: {args.dataset_choice}')	
	model = P3mNet()
	if torch.cuda.device_count()==0:
		print(f'Running on CPU...')
		args.cuda = False
		ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
	else:
		print(f'Running on GPU with CUDA as {args.cuda}...')
		ckpt = torch.load(args.model_path)
	model.load_state_dict(ckpt['state_dict'], strict=True)
	if args.cuda:
		model = model.cuda()
	if args.dataset_choice=='SAMPLES':
		test_samples(args,model)
	elif args.dataset_choice in ['P3M_500_P','P3M_500_NP']:
		logging_filename = TEST_LOGS_FOLDER+args.logname+'.log'
		if os.path.exists(logging_filename):
			os.remove(logging_filename)
		logging.basicConfig(filename=logging_filename, level=logging.INFO)
		args.logging = logging
		test_p3m10k(args, model)
	else:
		print('Please input the correct dataset_choice (SAMPLES, P3M_500_P or P3M_500_NP).')


if __name__ == '__main__':
	args = get_args()
	load_model_and_deploy(args)
