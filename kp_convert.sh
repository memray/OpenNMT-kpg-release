#!/usr/bin/env bash
BASE_DATA_DIR="data/keyphrase/json"
TOKENIZER="meng17"
OUTPUT_DIR="data/keyphrase/$TOKENIZER"

declare -a train_sets=("kp20k" "magkp" "kp20k_small")
declare -a valid_sets=("kp20k" "duc" "inspec" "krapivin" "nus" "semeval")
declare -a test_sets=("kp20k" "kp20k_small" "duc" "inspec" "krapivin" "nus" "semeval")

train_param="-filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -lower -shuffle -tokenizer $TOKENIZER -replace_digit"
valid_param="-lower -include_original -tokenizer $TOKENIZER -replace_digit"
test_param="-lower -include_original -tokenizer $TOKENIZER -replace_digit"

for train in "${train_sets[@]}"
do
    echo 'Processing train' $train
    cmd='-src_file '"$BASE_DATA_DIR"'/'"$train"'/'"$train"'_train.json -output_path '"$OUTPUT_DIR"'/'"$train"'/'"$train"'_train'
    python -m kp_data_converter $cmd $train_param
done

for valid in "${valid_sets[@]}"
do
    echo 'Processing valid' $valid
    cmd='-src_file '"$BASE_DATA_DIR"'/'"$valid"'/'"$valid"'_valid.json -output_path '"$OUTPUT_DIR"'/'"$valid"'/'"$valid"'_valid'
    python -m kp_data_converter $cmd $valid_param
done

for test in "${test_sets[@]}"
do
    echo 'Processing test' $test
    cmd='-src_file '"$BASE_DATA_DIR"'/'"$test"'/'"$test"'_test.json -output_path '"$OUTPUT_DIR"'/'"$test"'/'"$test"'_test'
    python -m kp_data_converter $cmd $test_param
done

#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_validation.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2many -include_original

#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_testing.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_test -lower -include_original -tokenizer meng17 -replace_digit

# a small KP20k for debugging
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_training_nodup.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train_small -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

# for MAG
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/magkp_training.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/magkp_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle

#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_validation.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -include_original

#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_testing.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_test -lower -include_original -tokenizer meng17 -replace_digit