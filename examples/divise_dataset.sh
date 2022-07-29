# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash
# datasets="KRAS_G12D MAT2A PDL1"
# datasets="ALK BRAF BTK CDK4 EGFR FGFR1 JAK2 NTRK1 VEGFR2"
dataset=$1
if [ ! -d "./data/$dataset" ]; then
      mkdir -p "./data/$dataset"
fi
for dataset in $dataset; do
      data_path="/home/jovyan/TankBind/examples/data/stonewise_rank_sos1_hpk1"
      # data_path="./SAR-interface/input"
      out_data_path="./data/$dataset"
      /opt/conda/bin/python \
      divise_dataset.py --dataset_name=$dataset \
				--data_path=$data_path  \
                        --out_data_path=$out_data_path  
done

# copy to target
# mkdir -p logits/$ckpt_name
# cp y_pred.pt logits/$ckpt_name/y_pred.pt
