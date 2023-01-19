# remember to pass the batch size to this model as first arg
# --accumulate_grad_batches=8
python main.py --limit_train_batches=0.03 --num_workers="$2" --accelerator=gpu --devices=auto --load_saved_dataloader --batch_size="$1" --save_frequency=33 --eval_frequency=33 --max_epochs=33 --log_every_n_steps=30 --default_root_dir=./save/logs