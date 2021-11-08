# Keyphrase Generation (built on OpenNMT-py)

This is a repository providing code and datasets used in [One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases](https://arxiv.org/abs/1810.05241) and [Does Order Matter? An Empirical Study on Generating Multiple Keyphrases as a Sequence](https://arxiv.org/abs/1909.03590).

All datasets and selected model checkpoints in the papers can be downloaded here ([data.zip](https://drive.google.com/open?id=1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP) and [models.zip](https://drive.google.com/open?id=18Pfs0ePAMl17kfjYRU_9HxYc0eUXet-_)). Unzip the file `data.zip and models.zip` and override the original `data/ and model/` folder. 

## Quickstart

All the config files used for training and evaluation can be found in folder `config/`.
For more examples, you can refer to scripts placed in folder `script/`.


### Preprocess the data

```bash
source kp_convert.sh # dump json to src/tgt files (OpenNMT format)
python preprocess.py -config config/preprocess/config-preprocess-keyphrase-kp20k.yml
```

### Train a One2Seq model with Diversity Mechanisms enabled

```bash
python train.py -config config/train/config-rnn-keyphrase-one2seq-diverse.yml
```

### Train a One2One model

```bash
python train.py -config config/train/config-rnn-keyphrase-one2one-stackexchange.yml
```

### Run generation and evaluation 

```bash
python kp_gen_eval.py -tasks pred eval report -config config/test/config-test-keyphrase-one2seq.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17-one2seq-kp20k-topmodels/ -output_dir output/meng17-one2seq-topbeam-selfterminating/meng17-one2many-beam10-maxlen40/ -testsets duc inspec semeval krapivin nus -gpu -1 --verbose --beam_size 10 --batch_size 32 --max_length 40 --onepass --beam_terminate topbeam --eval_topbeam
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
- [Rui Meng](https://github.com/memray/) (University of Pittsburgh)
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
