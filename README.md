# Keyphrase Generation (built on OpenNMT-py)

This is a repository providing code and datasets used in [One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases](https://arxiv.org/abs/1810.05241) and [Does Order Matter? An Empirical Study on Generating Multiple Keyphrases as a Sequence](https://arxiv.org/abs/1909.03590).

All datasets and selected model checkpoints in the papers can be downloaded here ([data.zip](https://drive.google.com/open?id=1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP) and [models.zip](https://drive.google.com/open?id=18Pfs0ePAMl17kfjYRU_9HxYc0eUXet-_)). Unzip the file `data.zip` and `models.zip` and override the original `data/ and model/` folder. 

## Update (Jan 2022)

Merged with OpenNMT v2 and integrated a new pre-processing pipeline. Now training/inference can directly load **JSON data** from disk, without any hassle of tokenization or conversion to tensor files. 
 - Paper datasets and DUC ([download](https://drive.google.com/file/d/1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP/view)): KP20k/Inspec/Krapivin/NUS/SemEval2010/DUC2001.
 - 4 large annotated datasets ([download](https://drive.google.com/file/d/1VoXr7pZqLUDBi0PPtbsvj6jv05hYtWdh/view?usp=sharing)): KP20k, OpenKP, KPTimes+JPTimes, StackExchange.

Some config examples can be of help for you to kick off:
 - [Configs](https://github.com/memray/OpenNMT-kpg-release/tree/master/script/transfer/train_fulldata) using RoBERTa subword tokenization. Vocab (including dict.txt/merges.txt/vocab.json/tokenizer.json) can be found [here](https://drive.google.com/file/d/1SM-8c2u3AV2-_71pjSlGVD8wyT7sv6vm/view?usp=sharing).
 - [Configs](https://github.com/memray/OpenNMT-kpg-release/tree/master/script/empirical_study/diverse) using word tokenization. Vocab (magkp20k.vocab.json, 50k most frequent words in KP20k and MagKP) can be found [here](https://drive.google.com/file/d/1MJcQeORQBmDdEEjdxmZMVijnB9dR7pWs/view?usp=sharing).

All shared resources are placed [here](https://drive.google.com/drive/folders/1nJL-LC0M8lXdDEl0ZRQMc_rcuvvKO5Hb?usp=sharing).


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
