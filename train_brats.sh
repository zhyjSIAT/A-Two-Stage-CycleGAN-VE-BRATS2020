#!/bin/bash
# sh train_brats.sh "ve"

export CUDA_VISIBLE_DEVICES==6

if [ "$1" = "vp" ]
then
    echo "================ run configs/vp/ddpm_continuous.py ================"
    python main.py \
        --config=configs/vp/ddpm_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "ve" ]
then
    echo "================ run configs/ve/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/ve/ncsnpp_continuous.py \
        --mode='train'  \
        --workdir=results
else
    echo "================ You must input one argument: ve or vp ================"
fi