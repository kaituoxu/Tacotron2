#!/bin/bash

# Created on 2019/01
# Author: Kaituo XU (NPU-ASLP)

wav_dir=/home/ktxu/workspace/data/LJSpeech-1.1
csv_file=data/train.csv
stage=1

# Training
id=0
use_cuda=1
epochs=500
# minibatch
batch_size=16
# optimizer
lr=1e-3
l2=1e-6
# save and load model
checkpoint=0
# log and visualize
print_freq=10
visdom=0
visdom_id="Taco2 training"

# Synthesis
eval_csv=data/cv10.csv
show_spect=1
show_attn=1
eval_use_cuda=0

# exp tag
tag="" # tag for managing experiments.

ngpu=1
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


if [ -z ${tag} ]; then
    basename=`basename $csv_file`
    suffix=$(echo $basename | cut -d . -f1)
    expdir=exp/train_epoch${epochs}_bs${batch_size}_lr${lr}_l2${l2}_${suffix}
else
    expdir=exp/train_${tag}
fi

if [ $stage -le 1 ]; then
    echo "Stage 1: Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        CUDA_VISIBLE_DEVICES="$id" \
        train.py \
        --train_dir $wav_dir \
        --train_csv $csv_file \
        --use_cuda $use_cuda \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --l2 $l2 \
        --save_folder $expdir \
        --checkpoint $checkpoint \
        --print_freq $print_freq \
        --visdom $visdom \
        --visdom_id "$visdom_id"
fi

if [ $stage -le 2 ]; then
    echo "Stage 2: Synthesising"
    out_dir=$expdir/synthesis
    ${decode_cmd} --gpu ${ngpu} ${out_dir}/synthesis.log \
        CUDA_VISIBLE_DEVICES="$id" \
        synthesis.py \
        --model_path $expdir/final.pth.tar \
        --csv_file $eval_csv \
        --out_dir $out_dir \
        --show_spect $show_spect \
        --show_attn $show_attn \
        --use_cuda $eval_use_cuda
fi
