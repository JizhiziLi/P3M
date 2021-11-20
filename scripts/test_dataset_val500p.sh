#/bin/bash
# nickname: your test logging file along with evaluation results 
# will be in the file `logs/test_logs/nickname.log`
# test_choice: [HYBRID/RESIZE] the test strategy you want to use
# we use RESIZE (1/2) as in the paper
# model_path: the pretrained model path
# test_result_dir: the path to save the predict results


dataset_choice='P3M_500_P'
test_choice='RESIZE'
model_path='models/pretrained/p3mnet_pretrained_on_p3m10k.pth'
nickname=test_val500p

python core/test.py \
     --cuda \
     --dataset_choice=$dataset_choice \
     --logname=$nickname \
     --model_path=$model_path\
     --test_choice=$test_choice \
     --test_result_dir=result/$nickname/ \