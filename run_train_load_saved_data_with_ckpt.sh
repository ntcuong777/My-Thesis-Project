# remember to pass the batch size to this model as first arg
# --accumulate_grad_batches=8
python main.py --checkpoint_file="$3" --num_workers="$2" --accelerator=gpu --devices=auto --load_saved_dataloader --batch_size="$1" --save_frequency=4 --eval_frequency=4 --max_epochs=100 --log_every_n_steps=50 --default_root_dir=./save/logs