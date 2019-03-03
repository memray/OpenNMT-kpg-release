#!/usr/bin/env bash
cd ~/project/kp/OpenNMT-kpg/

# for KP20k
#python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_training_nodup.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

#python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_validation.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2many -include_original

python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_testing.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_test -lower -target_type one2many -include_original

# a small KP20k for debugging
#python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_training_nodup.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train_small -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

# for MAG
python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp_raw/magkp_training.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/magkp_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_validation.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2many -include_original

python -m onmt.keyphrase.corpus_converter -src_file ~/project/kp/OpenNMT-kpg/data/keyphrase/kp20k_raw/kp20k_testing.json -save_data ~/project/kp/OpenNMT-kpg/data/keyphrase/magkp/kp20k_test -lower -target_type one2many -include_original