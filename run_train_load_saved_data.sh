# remember to pass the batch size to this model as first arg
# --accumulate_grad_batches=8
tmp_1=64
tmp_2=$1
accum=$((tmp_1 / tmp_2))
python main.py --accumulate_grad_batches="$accum" --num_workers="$2" --accelerator=gpu --devices=auto --load_saved_dataloader --batch_size="$1" --save_frequency=5 --eval_frequency=5 --max_epochs=30 --log_every_n_steps=30 --gradient_clip_val=1 --check_val_every_n_epoch=30 --default_root_dir=./save/logs