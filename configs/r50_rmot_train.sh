# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


PRETRAIN=./r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=default
python3 -m torch.distributed.launch --nproc_per_node=8 \
   --use_env main.py \
   --meta_arch rmot \
   --use_checkpoint \
   --dataset_file e2e_rmot \
   --epoch 100 \
   --with_box_refine \
   --lr_drop 50 \
   --lr 1e-4 \
   --lr_backbone 1e-5 \
   --pretrained ${PRETRAIN}\
   --output_dir exps/${EXP_DIR} \
   --batch_size 1 \
   --sample_mode random_interval \
   --sample_interval 1 \
   --sampler_steps 60 80 90 \
   --sampler_lengths 2 2 2 2 \
   --update_query_pos \
   --merger_dropout 0 \
   --dropout 0 \
   --random_drop 0.1 \
   --fp_ratio 0.3 \
   --query_interaction_layer QIM \
   --data_txt_path_train ./datasets/data_path/refer-kitti.train \
   --refer_loss_coef 2
