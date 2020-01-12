# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import copy
import string
from collections import Counter

import spacy
import re
import numpy as np

from onmt.newssum import fragutils
from tools.stanfordcorenlp import StanfordCoreNLP

from nltk.tokenize import sent_tokenize
import onmt.newssum.rouge_eval.rouge as rouge
metric_keys = ["rouge-1", "rouge-2", "rouge-l", "entity_cov"]

import nltk
nltk.data.path.append('/export/share/rmeng/tools/nltk')
stemmer = nltk.stem.porter.PorterStemmer()
stopword_set = set(nltk.corpus.stopwords.words('english'))
stopword_set.update(['\'s', 'doe', 'n\'t', 'and', 'also', 'whether'])
stanfordnlp = None
spacy_nlp = spacy.load('en_core_web_sm')

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def word_tokenize(text, model="spacy", lowercase=False):
    if model == 'stanfordnlp':
        global stanfordnlp
        if not stanfordnlp:
            stanfordnlp = StanfordCoreNLP('/export/share/rmeng/tools/stanford-corenlp/stanford-corenlp-full-2018-10-05/',
                                          memory='8g', port=6666)
        return stanfordnlp.word_tokenize(text)
    elif model == 'spacy':
        spacy_doc = spacy_nlp(text, disable=["tagger", "parser", "ner", "textcat"])
        tokens = [t.text.strip() for t in spacy_doc if len(t.text.strip()) > 0]
        tokens = [
            str(t).lower()
            if lowercase
            else str(t)
            for t in tokens
        ]
        return tokens
    else:
        raise NotImplementedError


def get_noun_chunks(text, trim_punct=True, remove_stopword=True):
    spacy_doc = spacy_nlp(text, disable=["textcat"])
    np_chunks = list(spacy_doc.noun_chunks)
    np_str_list = []
    for chunk in np_chunks:
        np = []
        for w in chunk:
            w = w.text
            if trim_punct:
                w = w.strip(r"""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~""")
            if remove_stopword:
                if w.lower() in stopword_set:
                    continue
            np.append(w)
        if len(np) > 0:
            np_str_list.append(np)
    np_chunks = [{'start': np.start, 'end': np.end, 'text': np.text, 'label_': np.label_} for np in np_chunks]

    return np_str_list, np_chunks


def get_NERs(text):
    spacy_doc = spacy_nlp(text, disable=["textcat"])
    ner_chunks = list(spacy_doc.ents)
    ner_str_list = []
    for chunk in ner_chunks:
        ner = []
        for w in chunk:
            w = w.text
            ner.append(w)
        if len(ner) > 0:
            ner_str_list.append(ner)

    ner_chunks = [{'start': ner.start, 'end': ner.end, 'text': ner.text, 'label_': ner.label_} for ner in ner_chunks]

    return ner_str_list, ner_chunks

def get_verbs(text):
    spacy_doc = spacy_nlp(text, disable=["parser", "ner", "textcat"])
    verb_tokens = [token for token in spacy_doc if token.pos_ == "VERB"]
    verb_tokens = [{'text': token.text, 'i': token.i, 'lemma_': token.lemma_, 'pos_': token.pos_} for token in verb_tokens]
    return verb_tokens


def build_matching_mask(text_tokens, match_tokens,
                        lowercase=False, stem=False,
                        remove_punct=False, remove_stopwords=False):
    def _preprocess(tokens, lowercase, stem, remove_punct, remove_stopwords):
        if lowercase:
            tokens = [w for w in tokens]
        if stem:
            tokens = [stemmer.stem(w) for w in tokens]
        if remove_punct:
            tokens = [w for w in tokens if w not in string.whitespace+string.punctuation]
        if remove_stopwords:
            tokens = [w for w in tokens if w.lower() not in stopword_set]

        return tokens

    match_set = set(_preprocess(match_tokens,
                                lowercase=lowercase, stem=stem,
                                remove_punct=remove_punct, remove_stopwords=remove_stopwords))
    processed_text_tokens = _preprocess(text_tokens, lowercase=lowercase, stem=stem,
                                        remove_punct=False, remove_stopwords=False)
    mask = []
    for t in processed_text_tokens:
        mask.append(1 if t in match_set else 0)

    assert len(mask) == len(text_tokens)
    return  mask


def build_bottomup_mask(src_sents, tgt_tokens):
    """
    Refactored from https://github.com/sebastianGehrmann/bottom-up-summary
    :param src_sents: a list of sentences, each sentence is a list of words
    :param tgt_tokens:
    :return:
    """
    def _compile_substring(start, end, split):
        if start == end:
            return split[start]
        return ' '.join(split[start: end + 1])

    # flatten sentences to a list, will be stacked back in the end
    src_tokens = [w for s in src_sents for w in s]
    tgt_str = (' '.join(tgt_tokens)).replace('\n', '[IGNORE]')
    startix = 0
    endix = 0
    matches = []
    match_fragments = []
    matchstrings = Counter()
    while endix < len(src_tokens):
        # last check is to make sure that phrases at end can be copied
        searchstring = _compile_substring(startix, endix, src_tokens)
        if searchstring in tgt_str and endix < len(src_tokens)-1:
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend([0] * (endix - startix + 1))
                endix += 1
            else:
                # if you want phrases not words, change if-condition to startix>=endix-1
                full_string = _compile_substring(startix, endix - 1, src_tokens)
                # if the same substring appears before, ignore it
                if matchstrings[full_string] >= 1:
                    matches.extend([0] * (endix - startix))
                else:
                    matches.extend([1] * (endix - startix))
                    matchstrings[full_string] += 1
                    fragment =  {'src_start': startix,
                                 'src_end': endix,
                                 'match_score': endix - startix,
                                 'words': src_tokens[startix: endix],
                                 'full_string': full_string,
                                 'src_sentid': None,}
                    match_fragments.append(fragment)
                #endix += 1
            startix = endix

    # convert mask back to seperate sentences
    sents_mask = []
    start_offset = 0
    for sent in src_sents:
        sents_mask.append(matches[start_offset: start_offset + len(sent)])
        start_offset += len(sent)

    # convert indexes in fragments
    src_sent_lengths = [len(s) for s in src_sents]
    accum_sent_lengths = [sum(src_sent_lengths[: sid + 1]) for sid, _ in enumerate(src_sent_lengths)]
    for fragment in match_fragments:
        sent_id = 0
        ori_start = fragment['src_start']
        ori_end = fragment['src_end']
        while ori_end > accum_sent_lengths[sent_id]:
            sent_id += 1
        # print(fragment['full_string'])
        # print(src_sents[sent_id])
        fragment['src_sentid'] = sent_id
        fragment['src_start'] = (ori_start - accum_sent_lengths[sent_id-1]) if sent_id > 0 else ori_start
        fragment['src_end'] = (ori_end - accum_sent_lengths[sent_id-1]) if sent_id > 0 else ori_end
        # print(src_tokens[ori_start: ori_end])
        # print(src_sents[sent_id][fragment['src_start']: fragment['src_end']])

    assert len(matches) == len(src_tokens)
    assert len(src_sents) == len(sents_mask)

    return match_fragments, sents_mask


def build_oracle_sentence_mask(src_sents, tgt_sents, metric='avg'):
    best_sent_ids = []

    for tgt_sent in tgt_sents:
        tgt_str = ' '.join(tgt_sent)
        best_score = 0.0
        best_sent_id = 0

        for src_sent_id, src_sent in enumerate(src_sents):
            if len(src_sent) < 3:
                continue
            src_str = ' '.join(src_sent)
            rouge_dict = calc_rouge(src_str, tgt_str)
            if metric == 'avg':
                score = np.mean([float(v) for v in rouge_dict.values()])
            else:
                score = rouge_dict[metric]
            if score > best_score and src_sent_id not in best_sent_ids:
                best_score = score
                best_sent_id = src_sent_id

        best_sent_ids.append(best_sent_id)

    sent_masks = [0] * len(src_sents)
    for best_sent_id in best_sent_ids:
        sent_masks[best_sent_id] = 1

    word_masks = []
    for src_sent_id, src_sent in enumerate(src_sents):
        if src_sent_id in best_sent_ids:
            word_masks.append([1] * len(src_sent))
        else:
            word_masks.append([0] * len(src_sent))

    assert len(sent_masks) == len(src_sents)

    extracted_str = ' '.join([w for sid in best_sent_ids for w in src_sents[sid]])
    tgt_str = ' '.join([w for s in tgt_sents for w in s])
    extracted_str = extracted_str.replace('\n', ' ')
    tgt_str = tgt_str.replace('\n', ' ')
    rouge_score = calc_rouge(extracted_str, tgt_str)

    return best_sent_ids, sent_masks, word_masks, extracted_str, rouge_score


def mask_to_text(sents_tokens, token_masks):
    tokens = []
    for sent, sent_mask in zip(sents_tokens, token_masks):
        for token, token_mask in zip(sent, sent_mask):
            if token_mask == 1:
                tokens.append(token)

    return ' '.join(tokens)


def summary_sents_to_text(tgt_sents):
    tgt_tokens = []
    for sent in tgt_sents:
        tgt_tokens.extend(sent)
        if sent[-1] != '.':
            tgt_tokens.append('.')
    tgt_str = ' '.join(tgt_tokens)
    return tgt_str


def fragments_to_mask(flat_fragments, src_sents):
    '''
    Given extracted fragments, generate corresponding source text masks
    :return:
    '''
    mask = [[0] * len(s) for s in src_sents]

    for f in flat_fragments:
        for i in range(f['src_start'], f['src_end'] + 1):
            mask[f['src_sentid']][i] = 1
        # print(f['match_score'])
        # print(f['words'])
        # print(src_sents[f['src_sentid']][f['src_start']: f['src_end'] + 1])
        words = []
        for i, m in enumerate(mask[f['src_sentid']]):
            if m:
                words.append(src_sents[f['src_sentid']][i])
        # print(words)

    return mask


def fragment_to_text(top_fragments_list, src_sents):
    ex_summary_text = []
    word_set = set()
    for sent_frags in top_fragments_list:
        if len(ex_summary_text) > 0:
            ex_summary_text.append(' . ')
        for f in sent_frags:
            sentid = f['src_sentid']
            for wordid in range(f['src_start'], f['src_end'] + 1):
                if '%d_%d' % (sentid, wordid) not in word_set:
                    ex_summary_text.append(src_sents[sentid][wordid])
                    word_set.add('%d_%d' % (sentid, wordid))

    ex_summary_text = ' '.join(ex_summary_text)

    # flat_fragments = [f for frags in top_fragments_list for f in frags if len(f) > 0]
    # ex_summary_text = ' . '.join([' '.join(src_sents[f['src_sentid']][f['src_start']: f['src_end'] + 1]) for f in flat_fragments])

    return ex_summary_text

def build_oracle_fragment_mask(src_sents, tgt_sents,
                               ignore_punct=True,
                               ignore_stopword=True,
                               stemming=True):
    """
    Iteratively find the best matching fragments
    for each summary sentence, return a segment from the most similar sentence
    :param src_sents:
    :param tgt_sents:
    :param match_method: ['lcs', 'word']
    :param smoothing_window: only useful for word-based matching
    :param extend_to_boundary:
    :return:
        extracted_fragments: a list of fragments corresponding to each summary sent,
        each fragment is a dict, containing {'src_sentid', 'src_start', 'src_end', 'frag_words', 'sum_start', 'sum_end'}
    """
    tgt_sents_to_match = copy.copy(tgt_sents)
    if ignore_punct:
        tgt_sents_to_match =  [[w if w not in string.punctuation else '[IGNORE]' for w in s] for s in tgt_sents_to_match]
    if ignore_stopword:
        tgt_sents_to_match = [[w if w not in stopword_set else '[IGNORE]' for w in s] for s in tgt_sents_to_match]

    top_fragments_list, all_fragments_list = fragutils.extract_multiple_fragments(tgt_sents_to_match, src_sents,
                                                          match_method='lcs', smoothing_window=5,
                                                          min_sent_len=2, min_match_words=2, density_threshold=0.3,
                                                          stemming=stemming,
                                                          max_depth=10, extend_to_boundary=False)

    ex_summary_text = fragment_to_text(top_fragments_list, src_sents)
    tgt_str = summary_sents_to_text(tgt_sents)
    # print('Summary  :' + tgt_str)
    # print('Extracted:' + ex_summary_text)
    rouge_score = calc_rouge(ex_summary_text, tgt_str)
    # print(rouge_score)

    top_flat_fragments = [f for frags in top_fragments_list for f in frags if len(f) > 0]
    top_fragment_masks = fragments_to_mask(top_flat_fragments, src_sents)
    all_flat_fragments = [f for frags in all_fragments_list for f in frags if len(f) > 0]
    all_filtered_fragments = []
    for f in all_flat_fragments:
        count = 0
        for w in f['words']:
            if w.lower() not in stopword_set:
                count += 1
        if count >= 1:
            all_filtered_fragments.append(f)
    # print('%d/%d' % (len(filtered_fragments), len(all_flat_fragments)))
    all_fragment_masks = fragments_to_mask(all_filtered_fragments, src_sents)

    return top_fragments_list, all_fragments_list, top_flat_fragments, all_filtered_fragments, top_fragment_masks, all_fragment_masks, ex_summary_text, rouge_score


def prepend_space_to_words(words):
    new_words = []

    # prepend a space for all non-head and non-punctuation words
    for word in words:
        if len(new_words) == 0 or (len(word) == 1 and word in string.punctuation + string.whitespace):
            new_words.append(word)
        else:
            new_words.append(' ' + word)

    return new_words


def words_to_subwords(tokenizer, words):
    all_subwords = []
    all_codes = []
    spaced_words = prepend_space_to_words(words)

    for word in spaced_words:
        subwords = tokenizer.tokenize(word, add_prefix_space=True)
        codes = tokenizer.convert_tokens_to_ids(subwords)

        all_subwords.extend(subwords)
        all_codes.extend(codes)

    return all_subwords, all_codes


def wordmasks_to_subwords(tokenizer, title_sents_words, oracle_masks):
    mask_keys = oracle_masks.keys()
    mask_values = oracle_masks.values()

    subword_sents = []
    code_sents = []
    mask_sents = {}
    for k in mask_keys: mask_sents[k]=[]

    for sent_mask_tuple in zip(title_sents_words, *list(mask_values)):
        sent = sent_mask_tuple[0]
        spaced_words = prepend_space_to_words(sent)
        sent_masks = sent_mask_tuple[1:]
        # print(spaced_words)
        # print(len(sent_masks))

        subword_sent = []
        code_sent = []
        mask_sent = {}
        for k in mask_keys: mask_sent[k]=[]

        for word_mask_tuple in list(zip(spaced_words, *sent_masks)):
            word = word_mask_tuple[0]
            mask = word_mask_tuple[1:]
            subwords = tokenizer.tokenize(word, add_prefix_space=True)
            codes = tokenizer.convert_tokens_to_ids(subwords)

            subword_sent.extend(subwords)
            code_sent.extend(codes)
            # print(subwords)
            # print(codes)

            for mid, mkey in enumerate(mask_keys):
                mask_sent[mkey].extend([mask[mid]] * len(codes))
                # print([mask[mid]] * len(codes))
            # pass

        subword_sents.append(subword_sent)
        code_sents.append(code_sent)
        for k,v in mask_sent.items():
            mask_sents[k].append(v)
        pass

    return subword_sents, code_sents, mask_sents

def sentence_split(text, model="spacy"):
    if model == "spacy":
        spacy_doc = spacy_nlp(text)
        segmented_sents = [sent.text for sent in spacy_doc.sents]
    elif  model == "nltk":
        segmented_sents = sent_tokenize(text)

    return segmented_sents


def calc_rouge(pred_sent, gt_sent, stopwords_removal=False, stemming=True, lowercase=True):
    if lowercase:
        pred_sent = pred_sent.lower()
        gt_sent = gt_sent.lower()
    rouge_metric = rouge.Rouge(stopwords_removal=stopwords_removal, stemming=stemming)
    if pred_sent == None or gt_sent == None \
            or len(pred_sent.strip()) == 0 or len(gt_sent.strip()) == 0:
        fscores = {k: 0.0 for k in metric_keys}
    else:
        try:
            scores = rouge_metric.get_scores(pred_sent, gt_sent)
            fscores = {k: v['f'] for k, v in scores[0].items()}
        except Exception:
            fscores = {k: 0.0 for k in metric_keys}

    if 'entity_cov' in fscores:
        del fscores['entity_cov']

    return fscores


def eval_rouge(doc, extract_sent_idx, number_to_cutoff=3, stopwords_removal=False, stemming=True, logger=None):
    rouge_metric = rouge.Rouge(stopwords_removal=stopwords_removal, stemming=stemming)
    extract_sent_idx = extract_sent_idx[: min(number_to_cutoff, len(extract_sent_idx))]
    # sort extracted sentences in the order of their appearance
    extract_sent_idx = sorted(extract_sent_idx)
    extracted_sents = [doc.get_sentences()[idx] for idx in extract_sent_idx if idx < len(doc.get_sentences())]
    hypothesis = ' '.join(extracted_sents)
    reference = doc.summary

    if hypothesis == None or reference == None or len(hypothesis.strip()) == 0 or len(reference.strip()) == 0:
        fscores = {k: 0.0 for k in metric_keys}
    else:
        scores = rouge_metric.get_scores(hypothesis, reference)
        fscores = {k: v['f'] for k, v in scores[0].items()}

    # if logger:
    #     logger.info("Scores: %s" % fscores)
    # else:
    #     print("Scores: %s" % fscores)

    return fscores


def extract_entities(sentence, return_text=False):
    parsed_doc = spacy_nlp(sentence)
    if return_text:
        entities = [ent.text for ent in parsed_doc.ents]
    else:
        entities = parsed_doc.ents

    return entities


def calc_entity_coverage(pred_entities, gt_entities):
    correct_entities = set(gt_entities) & set(pred_entities)
    entity_cov = float(len(correct_entities)) / len(gt_entities) if len(gt_entities) > 0 else 0.0
    return entity_cov


def eval_entity_coverage(doc, extract_sent_idx, number_to_cutoff=3, logger=None):
    extract_sent_idx = extract_sent_idx[: min(number_to_cutoff, len(extract_sent_idx))]
    extract_sent_idx = sorted(extract_sent_idx)
    extracted_sents = [doc.get_sentences()[idx] for idx in extract_sent_idx if idx < len(doc.get_sentences())]
    hypothesis = '. '.join(extracted_sents)
    reference = doc.summary

    parsed_doc = spacy_nlp(hypothesis)
    hypothesis_entities = set([ent.text.lower() for ent in parsed_doc.ents])
    parsed_doc = spacy_nlp(reference)
    reference_entities = set([ent.text.lower() for ent in parsed_doc.ents])

    if logger:
        print_fn=logger.info
    else:
        print_fn=print
    print_fn("-" * 50)
    print_fn("Hypothesis: \n\tSummary: %s" % hypothesis)
    print_fn("\t Entities in summary: %s" % hypothesis_entities)
    print_fn("Ground-truth: \n\tSummary: %s" % reference)
    print_fn("\t Entities in summary: %s" % reference_entities)
    correct_entities = hypothesis_entities & reference_entities
    entity_cov = float(len(correct_entities)) / len(reference_entities) if len(reference_entities) > 0 else 0.0
    print_fn("Recall: %d/%d=%s" % (len(correct_entities), len(reference_entities), "%.4f" % entity_cov if entity_cov != None else "N/A"))

    return {"entity_cov": entity_cov}


def print_hypothesis(doc, extract_sent_idx, extract_sent_scores=None, logger=None):
    for sent_num, extract_sent_id in enumerate(extract_sent_idx):
        score = str(extract_sent_scores[sent_num]) if extract_sent_scores else 'N/A'
        if logger:
            logger.info("\t[%d][index=%d][score=%s] %s" % (sent_num + 1, extract_sent_id, score, doc.text_sents[extract_sent_id]))
        else:
            print("\t[%d][index=%d][score=%s] %s" % (sent_num + 1, extract_sent_id, score, doc.text_sents[extract_sent_id]))


def update_score(score_dict, new_score_dict):
    for metric, score in new_score_dict.items():
        score_dict[metric].append(score)


def tokenize(text):
    """
    Tokenizes input using the fastest possible SpaCy configuration.
    This is optional, can be disabled in constructor.
    """
    tokens = spacy_nlp(text, disable=["tagger", "parser", "ner", "textcat"])
    return [t.text for t in tokens]


def normalize(tokens, case=False):
    """
    Lowercases and turns tokens into distinct words.
    """
    tokens = [
        str(t).lower()
        if not case
        else str(t)
        for t in tokens
    ]
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def clean_text(text):
    text = re.sub(r"\W", " ", text)
    return text

def summary_sentence_segment(summary, dataset='cnndm'):
    if dataset.lower() == 'cnndm' or dataset.lower() == 'cnn' or dataset.lower() == 'dailymail':
        return [s.strip() for s in summary.split('\n')]
    elif dataset.lower() == 'nyt' or dataset.lower() == 'newyorktimes':
        sents = []
        for s in summary.split(';'):
            s = s.rstrip('(MS) ')
            tokens = s.split()
            if len(tokens) < 2:
                continue
            if len(tokens) < 5 and (s.startswith('photo') or s.startswith('graph') or s.startswith('chart')
                                    or s.startswith('map') or s.startswith('table') or s.startswith('drawing')
                                    or s.startswith('excerpt') or s.endswith('photo') or s.endswith('photos')
                                    or s.endswith('column') or s.endswith('comments') or s.endswith('recipe')
                                    or s.endswith('recipes')):
                continue
            s = s.strip()
            sents.append(s)
        return sents
    elif dataset.lower() == 'newsroom':
        summary = summary.replace('%20', ' ')
        sents = []
        # embarrassingly, nltk works much better than spacy on noisy data
        for s in sentence_split(summary, model="nltk"):
            s = s.strip()
            tokens = s.split()
            if len(tokens) < 4:
                continue
            sents.append(s)
        return sents
    elif dataset.lower() == 'xsum':
        # per my observation it's all one-sent summary
        return [summary.strip()]
    else:
        raise NotImplementedError

def tokenize_source_summary(source, summary, title, dataset):
    tokenized_title = normalize(tokenize(title), case=True)
    # process source text
    if source:
        source_sents = sentence_split(source)
        tokenized_source_sents = [normalize(tokenize(s), case=True) for s in source_sents]
    else:
        source_sents, tokenized_source_sents = None, None

    # process summary text
    summary_sents = summary_sentence_segment(summary, dataset)
    tokenized_summary_sents = [normalize(tokenize(s), case=True) for s in summary_sents]

    return tokenized_source_sents, tokenized_summary_sents, tokenized_title


def preprocess_sents(sents, case=False, stemming=True,
                     ignore_punc=True, ignore_stopword=False):
    if ignore_punc:
        sents = [[t if t.lower() not in string.punctuation else '[IGNORE]' for t in s] for s in sents]
    if ignore_stopword:
        sents = [[t if t.lower() not in stopword_set else '[IGNORE]' for t in s] for s in sents]
    if not case:
        sents = [[t.lower() for t in s] for s in sents]
    if stemming:
        sents = [[stemmer.stem(t) for t in s] for s in sents]

    return sents


def concat_sents(sents):
    tokens = []
    for s in sents:
        if len(tokens) > 0:
            tokens.append('[SEP]')
        tokens.extend(s)
    return tokens