python run_squad.py \
 --model_type bert \
 --model_name_or_path bert-base-multilingual-cased \
 --output_dir models \
 --data_dir data \
 --train_file KorQuAD_v1.0_train.json \
 --predict_file KorQuAD_v1.0_dev.json \
 --evaluate_during_training \
 --per_gpu_train_batch_size 6 \
 --per_gpu_eval_batch_size 6 \
 --max_seq_length 384 \
 --learning_rate 3e-5 \
 --logging_steps 5000 \
 --save_steps 5000 \
 --num_train_epochs 2 \
 --do_train