#!/bin/bash
PREFIX='/workspace/phd/strangan-chase-2021/src'
DATA_PATH='/workspace/phd/strangan/data/preprocessed/opportunity_all_users.npz'
SAVE_PREFIX='/tmp/strangan-runs/'
RUN_ID='2'
SAVE_DIR=$SAVE_PREFIX$RUN_ID

python -W ignore $PREFIX/strangan.py -d $DATA_PATH \
  -ss 'S1' \
  -st 'S1' \
  -ps 'LUA' \
  -pt 'BACK' \
  -ch 3 \
  -cls 4 \
  -bs 32 \
  --n_epochs 20 \
  --gan_epochs 20 \
  --gpu 0 \
  --lr_FC 0.002 --lr_FC_b1 0.9 --lr_FC_b2 0.999 \
  --lr_FD 0.0002 \
  --lr_G 0.00002 --lr_G_b1 0.5 --lr_G_b2 0.999 \
  --gamma 0.9 \
  --save_dir $SAVE_DIR \
  --log_interval 50 \
  --eval_interval 500 \
  --seed 0 \

#  --clf_ckpt /tmp/strangan-runs/1/clf.pt
#  --gen_ckpt $SAVE_DIR/gen.pt \
#  --dsc_ckpt $SAVE_DIR/dsc.pt \
#  --clf_ckpt $SAVE_DIR/clf.pt \
#  --resume_gan 1

#,S2,S3,S4
#,S2,S3,S4
