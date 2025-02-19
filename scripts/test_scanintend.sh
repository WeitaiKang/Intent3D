# TORCH_DISTRIBUTED_DEBUG=INFO 
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 3100 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /data/kangweitai/3DIG/ \
    --val_freq 1 --batch_size 6 --print_freq 10 \
    --lr_backbone=1e-3 --lr=1e-4 --text_encoder_lr=1e-5 \
    --dataset scanintend --test_dataset scanintend \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align  \
    --log_dir /data/kangweitai/3DIG/output/IntentNet/ \
    --tensorboard /data/kangweitai/3DIG/tensorboard/IntentNet/ \
    --lr_decay_epochs 65 \
    --pp_checkpoint /data/kangweitai/3DIG/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 90 --eval_batch_size 8  --eval_val --num_target 50 \
    --checkpoint_path /data/kangweitai/3DIG/output/IntentNet/scanintend/1708736928/ckpt_best.pth