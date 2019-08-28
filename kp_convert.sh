#!/usr/bin/env bash
BASE_DATA_DIR="data/keyphrase/json"
TOKENIZER="meng17"
OUTPUT_DIR="data/keyphrase/$TOKENIZER"

declare -a train_sets=("kp20k" "magkp" "kp20k_small")
declare -a valid_sets=("kp20k" "inspec" "krapivin" "semeval")
declare -a test_sets=("kp20k" "kp20k_valid2k" "kp20k_valid500" "kp20k_small" "duc" "inspec" "krapivin" "nus" "semeval")
#declare -a train_sets=("stackexchange")
#declare -a valid_sets=("stackexchange")
#declare -a test_sets=("stackexchange")

train_param="-filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -lower -shuffle -tokenizer $TOKENIZER -replace_digit"
for train in "${train_sets[@]}"
do
    echo 'Processing train' $train
    cmd='python kp_data_converter.py -src_file '"$BASE_DATA_DIR"'/'"$train"'/'"$train"'_train.json -output_path '"$OUTPUT_DIR"'/'"$train"'/'"$train"'_train '"$train_param"
    echo $cmd
    eval $cmd
done

valid_param="-lower -include_original -tokenizer $TOKENIZER -replace_digit"
for valid in "${valid_sets[@]}"
do
    echo 'Processing valid' $valid
    cmd='python kp_data_converter.py -src_file '"$BASE_DATA_DIR"'/'"$valid"'/'"$valid"'_valid.json -output_path '"$OUTPUT_DIR"'/'"$valid"'/'"$valid"'_valid '"$valid_param"
    echo $cmd
    eval $cmd
done

# only test part is well tested
test_param="-lower -include_original -tokenizer $TOKENIZER -replace_digit"
for test in "${test_sets[@]}"
do
    echo 'Processing test' $test
    cmd='python kp_data_converter.py -src_file '"$BASE_DATA_DIR"'/'"$test"'/'"$test"'_test.json -output_path '"$OUTPUT_DIR"'/'"$test"'/'"$test"'_test '"$test_param"
    echo $cmd
    eval $cmd
done

#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_validation.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_valid -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2many -include_original
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_testing.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_test -lower -include_original -tokenizer meng17 -replace_digit

# a small KP20k for debugging
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_training_nodup.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train_small -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

# for MAG
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/magkp_training.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/magkp_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_validation.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -include_original
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/kp20k/kp20k_testing.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_test -lower -include_original -tokenizer meng17 -replace_digit

# for StackExchange
#python -m kp_data_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/json/stackexchange/stackexchange_train.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/meng17/stackexchange/stackexchange_train -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -lower -shuffle -tokenizer meng17 -replace_digit -is_stack
#python -m kp_data_converter -src_file ~/project/keyphrase/OpenNMT-kpg/data/kp/json/stackexchange/stackexchange_valid.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/meng17/stackexchange/stackexchange_valid -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -lower -include_original -tokenizer meng17 -replace_digit -is_stack
#python -m kp_data_converter -src_file ~/project/keyphrase/OpenNMT-kpg/data/kp/json/stackexchange/stackexchange_test.json -output_path ~/project/kp/OpenNMT-kpg/data/keyphrase/meng17/stackexchange/stackexchange_test -lower -include_original -tokenizer meng17 -replace_digit -is_stack
