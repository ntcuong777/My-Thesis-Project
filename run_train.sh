# remember to pass the batch size to this model as first arg, num_workers as 2nd flag
#--accumulate_grad_batches=8
python main.py --num_workers="$2" --accelerator=gpu --devices=auto --batch_size="$1" --save_frequency=4 --eval_frequency=4 --max_epochs=20 --log_every_n_steps=30 --default_root_dir=./save/logs