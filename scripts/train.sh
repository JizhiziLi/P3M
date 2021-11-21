#/bin/bash

batchsizePerGPU=8
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
threads=8
nEpochs=150
lr=0.00001
nickname=train

python core/train.py \
    --logname=$nickname \
    --batchSize=$batchsize \
    --threads=$threads \
    --nEpochs=$nEpochs \
    --lr=$lr \
    --model_save_dir=models/trained/$nickname/ \