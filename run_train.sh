# remember to pass the batch size to this model as first arg, num_workers as 2nd flag
#--accumulate_grad_batches=8
tmp_1=64
tmp_2=$1
accum=$((tmp_1 / tmp_2))
python main.py --accumulate_grad_batches="$accum" --num_workers="$2" --accelerator=gpu --devices=auto --batch_size="$1" --save_frequency=5 --eval_frequency=5 --max_epochs=40 --log_every_n_steps=30 --default_root_dir=./save/logs