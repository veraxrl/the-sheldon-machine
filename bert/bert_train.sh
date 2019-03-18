python3 bert_classifier.py \
--data_dir=data \
--bert_model=bert-base-uncased \
--task_name=cola \
--output_dir=bert_output \
--do_train \
--do_eval \
--max_seq_length=128 \
--train_batch_size=16 \
--num_train_epochs=3.0 \