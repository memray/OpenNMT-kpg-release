# Keyphrase Generation (built on OpenNMT-py)

This is a repository providing code and datasets for keyphrase generation.

## Update (October 2022)
Release the resources of paper [**General-to-Specific Transfer Labeling for Domain Adaptable Keyphrase Generation**](https://arxiv.org/pdf/2208.09606.pdf). All scores can be found [here](https://docs.google.com/spreadsheets/d/1AWUdVbOsOn_F6rGeHm-xOBK6DTQ4LqRAaiT-QIAhFmk/edit?usp=sharing).

All datasets and selected model checkpoints in the papers can be downloaded from Huggingface Hub ([data](https://huggingface.co/datasets/memray/keyphrase/tree/main) and [ckpt](https://huggingface.co/memray/opennmt-kpg/tree/main)).
Config files can be found at [script/](https://github.com/memray/OpenNMT-kpg-release/tree/master/script/transfer/train_fulldata). 

For example, you can start training a Transformer model on KP20k using OpenNMT:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -config config/transfer_kp/train/transformer-presabs-kp20k.yml
```
To train a BART model on KP20k using [fairseq-kpg](https://github.com/memray/fairseq-kpg), vocab can be downloaded [here](https://huggingface.co/memray/opennmt-kpg/blob/main/roberta-base-kp.zip):
```bash
cd $FAIRSEQ_DIR
CUDA_VISIBLE_DEVICES=0 python train.py data/kp/json/kp20k/ --save-dir exps/kp/bartFT_presabs_kp20k_100k_rerun/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 128 --kp-concat-type pres_abs --arch bart_large --restore-file cache/bart.large/model.pt --bpe hf_pretrained_bpe --bpe-vocab hf_vocab/roberta-base-kp/vocab.json --bpe-merges hf_vocab/roberta-base-kp/merges.txt --dict-path hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-08 --clip-norm 0.1 --lr 1e-5 --update-freq 8 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 1024 --save-interval-updates 5000 --warmup-updates 10000 --total-num-update 100000 --num-workers 4 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project kp-project
```


## Update (April 2022)
Several pretrained checkpoints are available at Huggingface model repos.
- BART-large-KPG model pretrained with wikipedia phrases: [https://huggingface.co/memray/bart_wikikp](https://huggingface.co/memray/bart_wikikp/tree/main)
- BART/Transformers trained on four different keyphrase datasets: [https://huggingface.co/memray/opennmt-kpg/](https://huggingface.co/memray/opennmt-kpg/tree/main)

Two examples:
- Run kpg inference with Huggingface Transformers
```bash
CUDA_VISIBLE_DEVICES=0 python onmt/keyphrase/run_infer_hfkpg.py --config_name memray/bart_wikikp --model_name_or_path memray/bart_wikikp --tokenizer_name memray/bart_wikikp --dataset_name midas/duc2001 --do_predict --output_dir kp_output/duc2001/ --overwrite_output_dir --per_device_eval_batch_size 8 --predict_with_generate --text_column document --keyphrase_column extractive_keyphrases --source_prefix <present>10<header>5<category>5<seealso>2<infill>0<s> --num_beams 5 --generation_max_length 60
```
- Run kpg inference with OpenNMT-kpg (parameters are hard-coded)
```bash
CUDA_VISIBLE_DEVICES=0 python onmt/keyphrase/kpg_example_hfdatasets.py
```

## Update (Jan 2022)

Merged with OpenNMT v2 and integrated a new pre-processing pipeline. Now training/inference can directly load **JSON data** from disk, without any hassle of tokenization or conversion to tensor files.
Please check out Huggingface [repo](https://huggingface.co/datasets/memray/keyphrase/) for all resources.
~~- Paper datasets and DUC: KP20k/Inspec/Krapivin/NUS/SemEval2010/DUC2001.~~
~~- 4 large annotated datasets: KP20k, OpenKP, KPTimes+JPTimes, StackExchange.~~ 

Some config examples can be of help for you to kick off:
- [Configs](https://github.com/memray/OpenNMT-kpg-release/tree/master/script/transfer/train_fulldata) using RoBERTa subword tokenization. Vocab (including merges.txt/vocab.json/tokenizer.json) can be found [here](https://huggingface.co/memray/opennmt-kpg/blob/main/roberta-base-kp.zip).
- [Configs](https://github.com/memray/OpenNMT-kpg-release/tree/master/script/empirical_study/diverse) using word tokenization. Vocab (magkp20k.vocab.json, 50k most frequent words in KP20k and MagKP) can be found [here](https://huggingface.co/memray/opennmt-kpg/blob/main/magkp20k.vocab.json).

Please note that hf_vocab.tar.gz contains the vocab of subword tokenization (RoBERTa vocab with some new special tokens such as <SEP>), and [magkp20k.vocab.json](https://huggingface.co/memray/opennmt-kpg/blob/main/magkp20k.vocab.json) is for previous word tokenization based models (top 50k frequent words in magcs and kp20k).


## Quickstart

All the config files used for training and evaluation can be found in folder `config/`.
For more examples, you can refer to scripts placed in folder `script/`.


### Train a One2Seq model

```bash
python train.py -config config/transfer_kp/train/transformer-presabs-kp20k.yml
```

### Train a One2One model

```bash
python train.py -config config/transfer_kp/train/transformer-one2one-kp20k.yml
```

### Run generation and evaluation

```bash
# beam search (beamwidth=50)
python kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred eval -data_dir kp/data/kp/json/ -exp_root_dir kp/exps/transformer_exp_devbest/ -gpu 0 -batch_size 16 -beam_size 50 -max_length 40 -testsets kp20k openkp kptimes jptimes stackex kp20k_valid2k openkp_valid2k kptimes_valid2k jptimes_valid2k stackex_valid2k duc -splits test --data_format jsonl -gpu 0

# greedy decoding (beamwidth=1)
python kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred eval -data_dir kp/data/kp/json/ -exp_root_dir kp/exps/transformer_exp_devbest/ -gpu 0 -batch_size 16 -beam_size 1 -max_length 40 -testsets kp20k openkp kptimes jptimes stackex kp20k_valid2k openkp_valid2k kptimes_valid2k jptimes_valid2k stackex_valid2k duc -splits test --data_format jsonl -gpu 0
```

## Evaluation and Datasets
You may refer to `notebook/json_process.ipynb` to have a glance at the pre-processing.

We follow the data pre-processing and evaluation protocols in [Meng et al. 2017](https://arxiv.org/pdf/1704.06879.pdf). We pre-process both document texts and ground-truth keyphrases, including word segmentation, lowercasing and replacing all digits with symbol \<digit\>.

We manually clean the data examples in the valid/test set of KP20k (clean noisy text, replace erroneous keyphrases with actual author keyphrases, remove examples without any ground-truth keyphrases) and use scripts to remove invalid training examples (without any author keyphrase).

We evaluate models' performance on predicting present and absent phrases separately. Specifically, we first tokenize, lowercase and stem (using the Porter Stemmer of [NLTK](https://www.nltk.org/api/nltk.stem.html\#module-nltk.stem.porter)) the text, then we determine the presence of each ground-truth keyphrase by checking whether its words can be found verbatim in the source text.

To evaluate present phrase performance, we compute Precision/Recall/F1-score for each document taking only present ground-truth keyphrases as target and ignore the absent ones. We report the macro-averaged scores over documents that have at least one present ground-truth phrases (corresponding to the column \#PreDoc in the Table below, and similarly to the case of absent phrase evaluation.


![metrics](images/metric_formula.gif "metrics")

where #(pred) and #(target) are the number of predicted and ground-truth keyphrases respectively; and #(correct@k) is the number of correct predictions among the first k results.


We clarify that, since our study mainly focuses on keyword/keyphrase extraction/generation on short text, we only used the abstract of Semeval and NUS as source text. Therefore statistics like #PreKP may be different from the ones computed with fulltext, which also affect the final F1-scores. For the ease of reproduction, we post the detailed statistics in the following table and processed testsets with present/absent phrases split can be found in the released data (e.g. `data/json/kp20k/kp20k_test_meng17token.json`).


| **Dataset** | **#Train** | **#Valid** | **#Test** | **#KP** | **#PreDoc** | **#PreKP** | **#AbsDoc** | **#AbsKP** |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: 
| **KP20k** | 514k | 19,992 | 19,987 | 105,181 | 19,048 | 66,595 | 16,357 | 38,586|
| **Inspec** | -- | 1,500 | 500| 4,913 | 497 | 3,858 | 381 | 1,055 |
| **Krapivin** | -- | 1,844 | 460 | 2,641 | 437 | 1,485 | 417 | 1,156 |
| **NUS** | -- | - | 211 | 2,461 | 207 | 1,263 | 195 | 1,198 |
| **Semeval** | -- | 144 | 100 | 1,507 | 100 | 671 | 99 | 836|
| **StackEx** | 298k | 16,000 | 16,000 | 43,131 | 13,475 | 24,809 | 10,984 | 18,322 |
| **DUC** | -- | -- | 308 | 2,484 | 308 | 2,421 | 38 | 63 |




## Contributers
Major contributors are:
- [Rui Meng](https://github.com/memray/) (Salesforce Research, previously at University of Pittsburgh)
- [Eric Yuan](https://github.com/xingdi-eric-yuan) (Microsoft Research, Montréal)
- [Tong Wang](https://github.com/wangtong106) (Microsoft Research, Montréal)
- [Khushboo Thaker](https://github.com/khushsi) (University of Pittsburgh)


## Citation

Please cite the following papers if you are interested in using our code and datasets.
```
@article{meng2022general2specific,
  title={General-to-Specific Transfer Labeling for Domain Adaptable Keyphrase Generation},
  author={Meng, Rui and Wang, Tong and Yuan, Xingdi and Zhou, Yingbo and He, Daqing},
  journal={arXiv preprint arXiv:2208.09606},
  year={2022}
}
```
```
@inproceedings{meng2021empirical,
  title={An Empirical Study on Neural Keyphrase Generation},
  author={Meng, Rui and Yuan, Xingdi and Wang, Tong and Zhao, Sanqiang and Trischler, Adam and He, Daqing},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={4985--5007},
  year={2021}
}
```
```
@article{yuan2018onesizenotfit,
  title={One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases},
  author={Yuan, Xingdi and Wang, Tong and Meng, Rui and Thaker, Khushboo and He, Daqing and Trischler, Adam},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  url={https://arxiv.org/pdf/1810.05241.pdf},
  year={2020}
}
```
```
@article{meng2019ordermatters,
  title={Does Order Matter? An Empirical Study on Generating Multiple Keyphrases as a Sequence},
  author={Meng, Rui and Yuan, Xingdi and Wang, Tong and Brusilovsky, Peter and Trischler, Adam and He, Daqing},
  journal={arXiv preprint arXiv:1909.03590},
  url={https://arxiv.org/pdf/1909.03590.pdf},
  year={2019}
}
```
```
@inproceedings{meng2017kpgen,
  title={Deep keyphrase generation},
  author={Meng, Rui and Zhao, Sanqiang and Han, Shuguang and He, Daqing and Brusilovsky, Peter and Chi, Yu},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={582--592},
  url={https://arxiv.org/pdf/1704.06879.pdf},
  year={2017}
}
```
