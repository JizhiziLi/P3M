#/bin/bash
# model_path: the pretrained model path
# test_choice: [HYBRID/RESIZE] the test strategy you want to use
# We recommand HYBRID as for samples inference.

dataset_choice='SAMPLES'
test_choice='HYBRID'
model_path='models/pretrained/p3mnet_pretrained_on_p3m10k.pth'

python core/test.py \
     --cuda \
     --dataset_choice=$dataset_choice \
     --model_path=$model_path\
     --test_choice=$test_choice \

