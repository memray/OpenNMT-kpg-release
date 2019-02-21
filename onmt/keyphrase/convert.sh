#!/usr/bin/env bash
cd /Users/memray/Project/keyphrase/OpenNMT-kpg/

#python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_training_nodup.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2one

python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_training_nodup.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

#python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_validation.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2one -include_original

python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_validation.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_valid -lower -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -target_type one2many -include_original

python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_validation.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_test -lower -target_type one2many -include_original


#python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_training_nodup.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train_small -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2one
#python -m onmt.keyphrase.corpus_converter -src_file ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_training_nodup.json -save_data ~/Project/keyphrase/OpenNMT-kpg/data/keyphrase/kp20k/kp20k_train_small -lower -filter -max_src_seq_length 1000 -min_src_seq_length 10 -max_tgt_seq_length 8 -min_src_seq_length 1 -shuffle -target_type one2many

