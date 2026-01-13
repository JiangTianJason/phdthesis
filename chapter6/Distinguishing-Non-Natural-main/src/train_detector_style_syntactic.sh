###Use this script to start the experiment instead of the original "train_detector.sh" script
#################################################################

# 1st stage

## 2. form train set containing artificial（those attack failed）and origianl samples：
#python change_label_for_tsv.py \
#--input_file "../../embed_result/BadActs_embed_result/synbkd/yelp/train_poison.tsv" \
#--output_file data_corresponding/yelp/randomized_data_synbkd/train_randomized.tsv \
#--change_to 1 \
#--failed_flag Failed
#
#python change_label_for_tsv.py \
#--input_file "../../BadActs-master/poison_data/yelp/1/synbkd/train-clean.csv" \
#--output_file data_corresponding/yelp/randomized_data_synbkd/train_original.tsv \
#--adversarial_successful_failed_file data_corresponding/yelp/randomized_data_synbkd/train_randomized.tsv \
#--change_to 0 \
#--failed_flag Failed
#
#python concatenate.py \
#--input_file_1 data_corresponding/yelp/randomized_data_synbkd/train_randomized.tsv \
#--input_file_2 data_corresponding/yelp/randomized_data_synbkd/train_original.tsv \
#--output_file data_corresponding/yelp/randomized_data_synbkd/train.tsv


## 3. form test set containing artificial and origianl samples：
#python change_label_for_tsv_test_poison_clean.py \
#--input_file "../../embed_result/BadActs_embed_result/stylebkd/agnews/test_poison.tsv" \
#--output_file data_corresponding/agnews/randomized_data/test_randomized.tsv \
#--change_to 1 \
#--failed_flag Failed
#
#python change_label_for_tsv_test_poison_clean.py \
#--input_file "../../BadActs-master/poison_data/agnews/0/stylebkd/test-clean.csv" \
#--output_file data_corresponding/agnews/randomized_data/test_original.tsv \
#--adversarial_successful_failed_file data_corresponding/agnews/randomized_data/test_randomized.tsv \
#--change_to 0 \
#--failed_flag Failed
#
#python concatenate.py \
#--input_file_1 data_corresponding/agnews/randomized_data/test_randomized.tsv \
#--input_file_2 data_corresponding/agnews/randomized_data/test_original.tsv \
#--output_file data_corresponding/agnews/randomized_data/test.tsv


## 4. train the detector to distinguish between original and artificial samples:
#python run_detector_classification.py \
#--task_name agnews \
#--max_seq_len 128 \
#--do_train \
#--do_eval \
#--data_dir data/agnews/randomized_data \
#--output_dir experiments_detector/agnews/randomization \
#--model_name_or_path "bert-base-uncased" \
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 16 \
#--learning_rate 3e-5 \
#--num_train_epochs 5 \
#--svd_reserve_size 0 \
#--evaluation_strategy epoch \
#--overwrite_output_dir



### 2nd stage
## 2. form train set containing adversarial（those attack successful） and original samples:
#python form_dataset_detector.py \
#	--abnormal_file "../../embed_result/BadActs_embed_result/synbkd/agnews/train_poison.tsv" \
#	--abnormal_file_type tsv \
#	--normal_file "../../BadActs-master/poison_data/agnews/0/synbkd/train-clean.csv" \
#	--normal_file_type tsv \
#	--output_file data_corresponding_detection/agnews/synbkd/train.tsv \
#	--output_file_type tsv

## 3. form test set containing adversarial and original samples:
python form_dataset_detector_test_poison_clean.py \
	--abnormal_file "../../embed_result/BadActs_embed_result/stylebkd/agnews/test_poison.tsv" \
	--abnormal_file_type tsv \
	--abnormal_number 500 \
	--normal_file "../../BadActs-master/poison_data/agnews/0/stylebkd/test-clean.csv" \
	--normal_file_type tsv \
	--normal_number 500 \
	--output_file data_corresponding_detection/agnews/stylebkd/test_non_shuffle.tsv \
	--output_file_type tsv

## 4. train the detector to distinguish between original and adversarial samples:
#python run_detector_classification.py \
#--task_name agnews \
#--max_seq_len 128 \
#--do_train \
#--do_eval \
#--data_dir data_detection/agnews/stylebkd/ \
#--output_dir experiments_detector/agnews/stylebkd \
#--model_name_or_path experiments_detector/agnews/randomization \
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 16 \
#--learning_rate 3e-5 \
#--num_train_epochs 5 \
#--svd_reserve_size 0 \
#--evaluation_strategy epoch \
#--overwrite_output_dir