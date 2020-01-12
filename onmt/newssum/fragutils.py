# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import copy
from collections import defaultdict
import numpy as np

import nltk
nltk.data.path.append('/export/share/rmeng/tools/nltk')
stemmer = nltk.stem.porter.PorterStemmer()

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def extend_fragment_to_boundary(tokenized_sent, lend, rend):
    while lend > 0 and \
            tokenized_sent[lend] not in [',', '.', '!', '?', '-']:
        lend -= 1
    while rend < len(tokenized_sent) and \
            tokenized_sent[rend] not in [',', '.', '!', '?', '-']:
        rend += 1
    return lend, rend


def obtain_matching_fragments(summary_sent, src_sents,
                              match_method, smoothing_window,
                              min_frag_len, min_match_words, density_threshold, stemming
                              ):
    """

    :param summary_sent:
    :param src_sents:
    :param match_method:
    :param smoothing_window:
    :param min_frag_len:
    :param min_match_words:
    :param density_threshold: defined as #(match_words)/len(subsequence), 0 means return all results
    :return:
    """
    if match_method == 'word':
        matching_fragment_tuples = extract_fragments_by_wordmatch(summary_sent, src_sents,
                                                            smoothing_window=smoothing_window
                                                            )
    elif match_method == 'lcs':
        matching_fragment_tuples = extract_fragments_by_LCS(summary_sent, src_sents, stemming)

    filered_fragments = []
    for frag_tuple in matching_fragment_tuples:
        frag = frag_tuple[1]
        if frag['match_score'] >= min_match_words and (frag['src_end']-frag['src_start'] + 1 >= min_frag_len) \
            and frag['match_score']/(frag['src_end']-frag['src_start'] + 1) >= density_threshold:
            frag['src_sentid'] = frag_tuple[0]
            filered_fragments.append(frag)

    top_fragment = filered_fragments[0] if len(filered_fragments) > 0 else None

    return top_fragment, filered_fragments


def extract_fragments_iterative(summary_sent_id, summary_sent, src_sents,
                                match_method, smoothing_window,
                                min_frag_len, min_match_words, density_threshold, stemming,
                                cur_depth, max_depth,
                                extend_to_boundary=False):
    # obtain the primary fragment
    cent_best_frag, cent_frags = obtain_matching_fragments(summary_sent, src_sents,
                                         match_method, smoothing_window,
                                         min_frag_len, min_match_words, density_threshold, stemming
                                         )

    if extend_to_boundary:
        lend, rend = extend_fragment_to_boundary(src_sents[cent_best_frag['src_sentid']], cent_best_frag['src_start'], cent_best_frag['src_end'])
        cent_best_frag['src_start'] = lend
        cent_best_frag['src_end'] = rend

    # in case that no matching source sentence (mostly due to a noisy sentence in the groundtruth)
    if cent_best_frag is None:
        return [], []

    # obtain fragments from left remaining part
    top_fragments = []
    all_fragments = []

    if cent_best_frag['sum_start'] >= min_frag_len and cur_depth < max_depth:
        left_summary_sent = copy.copy(summary_sent)
        left_summary_sent = [t if tid < cent_best_frag['sum_start'] else '[IGNORE]' for tid, t in
                             enumerate(left_summary_sent)]
        left_best_frags, left_frags = extract_fragments_iterative(summary_sent_id, left_summary_sent, src_sents,
                                                 match_method, smoothing_window,
                                                 min_frag_len, min_match_words, density_threshold, stemming,
                                                 cur_depth + 1, max_depth)
    else:
        left_best_frags, left_frags = [], []

    top_fragments.extend(left_best_frags)
    all_fragments.extend(left_frags)
    top_fragments.append(cent_best_frag)
    all_fragments.extend(cent_frags)

    # obtain fragments from right remaining part
    if len(summary_sent) - cent_best_frag['sum_end'] - 1 >= min_frag_len and cur_depth < max_depth:
        right_summary_sent = copy.copy(summary_sent)
        right_summary_sent = [t if tid > cent_best_frag['sum_end'] else '[IGNORE]' for tid, t in
                              enumerate(right_summary_sent)]
        right_best_frags, right_frags = extract_fragments_iterative(summary_sent_id, right_summary_sent, src_sents,
                                                  match_method, smoothing_window,
                                                  min_frag_len, min_match_words, density_threshold, stemming,
                                                  cur_depth + 1, max_depth)
    else:
        right_best_frags, right_frags = [], []

    top_fragments.extend(right_best_frags)
    all_fragments.extend(right_frags)

    return top_fragments, all_fragments


def extract_multiple_fragments(tgt_sents, src_sents,
                               match_method='lcs', smoothing_window=5,
                               min_sent_len=2, min_match_words=2, density_threshold=0.5,
                               stemming=False,
                               max_depth=10, extend_to_boundary=False
                               ):
    """
    Iteratively find the best matching fragments
    for each summary sentence, return a segment from the most similar sentence
    :param tgt_sents: tokenized summary sentences
    :param src_sents: tokenized source sentences
    :param smoothing_window:
    :param min_sent_len: minimum length of each fragment
    :param min_match_words: minimum number of matching words in each fragment
    :param density_threshold: defined as #(match_words)/len(subsequence), 0 means return all results
    :param max_depth:
    :param extend_to_boundary:
    :return:
        extracted_fragments: a list of fragments corresponding to each summary sent,
            each fragment is a dict, containing {'src_sentid', 'src_start', 'src_end', 'frag_words', 'sum_start', 'sum_end'}
    """
    top_fragments_list = []
    all_fragments_list = []
    for summary_sent_id, summary_sent in enumerate(tgt_sents):
        top_fragments, all_fragments = extract_fragments_iterative(summary_sent_id, summary_sent, src_sents, match_method,
                                                smoothing_window, min_sent_len, min_match_words, density_threshold,
                                                stemming=stemming,
                                                cur_depth=1, max_depth=max_depth,
                                                extend_to_boundary=extend_to_boundary)
        # ex_summary_text = '\n'.join(
        #     [' '.join(src_sents[f['src_sentid']][f['src_start']: f['src_end'] + 1]) for f in top_fragments])
        # print(ex_summary_text)

        top_fragments_list.append(top_fragments)
        all_fragments_list.append(all_fragments)

    return top_fragments_list, all_fragments_list


def extract_singleton_fragments(sum_sents, src_sents,
                                match_method='word',
                                smoothing_window=5,
                                extend_to_boundary=False):
    """
    basically based on the extractive methods by Towards Annotating and Creating Summary Highlights at Sub-sentence Level
    for each summary sentence, return a segment from the most similar sentence
    :param sum_sents:
    :param src_sents:
    :param case:
    :param stemming:
    :param ignore_punc:
    :param smoothing_window:
    :param extend_to_boundary:
    :return:
        extracted_fragments: a list of fragments corresponding to each summary sent,
            each fragment is a dict, containing {'src_sentid', 'src_start', 'src_end', 'frag_words', 'sum_start', 'sum_end'}
    """
    extracted_fragments = []
    for summary_sent_id, summary_sent in enumerate(sum_sents):
        if match_method == 'word':
            matching_fragments = extract_fragments_by_wordmatch(summary_sent, src_sents,
                                                                smoothing_window = smoothing_window
                                                                )
        elif match_method == 'lcs':
            matching_fragments = extract_fragments_by_LCS(summary_sent, src_sents)
        # in case that no matching source sentence (mostly due to a noisy sentence in the groundtruth)
        best_sent_id, best_fragment = matching_fragments[0]
        if best_fragment is None:
            print(summary_sent)
            continue
        if extend_to_boundary:
            lend, rend = extend_fragment_to_boundary(src_sents[best_sent_id], best_fragment['src_start'], best_fragment['src_end'])
            best_fragment['src_start'] = lend
            best_fragment['src_end'] = rend

        best_fragment_words = src_sents[best_sent_id][best_fragment['src_start']: best_fragment['src_end'] + 1]
        best_fragment['src_sentid'] = best_sent_id
        best_fragment['frag_words'] = best_fragment_words
        best_fragment['sum_sentid'] = summary_sent_id

        extracted_fragments.append(best_fragment)

    return extracted_fragments


def locate_sum_fragment(sum_sent, src_sent, src_start, src_end):
    src_set = set(src_sent[src_start: src_end + 1])
    src_start, src_end = None, -1
    for wid, w in enumerate(sum_sent):
        if w in src_set:
            if src_start is None:
                src_start = wid
            if wid > src_end:
                src_end = wid
    return src_start, src_end


def smooth_fragments(fragments, smoothing_window):
    # stop until no more fragments can be merged
    while True:
        merged = False
        new_fragments = []
        fid = 0
        while fid < len(fragments):
            if fid == len(fragments) - 1:
                new_fragments.append(fragments[fid])
                fid += 1
            else:
                # check if the (fid+1) can be merged
                cur_fragment = fragments[fid]
                next_fragment = fragments[fid + 1]
                if next_fragment['src_start'] - cur_fragment['src_end'] - 1 <= smoothing_window:
                    new_fragment = {'src_start': cur_fragment['src_start'],
                                    'src_end': next_fragment['src_end'],
                                    'match_score': cur_fragment['match_score'] + next_fragment['match_score']}
                    new_fragments.append(new_fragment)
                    fid += 2
                    merged = True
                else:
                    new_fragments.append(cur_fragment)
                    fid += 1

        fragments = new_fragments
        if not merged:
            break

    return fragments


def _lcs(sum_sent, src_sent, src_sent_id, stemming):
    val_mat = np.zeros(shape=(len(sum_sent), len(src_sent)))
    # store triplets of (sum_start, sum_end, src_start, src_end, match_score)
    idx_mat = [[None] * len(src_sent)] * len(sum_sent)

    if stemming:
        src_sent = [stemmer.stem(t) for t in src_sent]
        sum_sent = [stemmer.stem(t) for t in sum_sent]

    for sum_t_id, sum_t in enumerate(sum_sent):
        for src_t_id, src_t in enumerate(src_sent):
            cur_val = 1 if src_t in sum_t == src_t else 0
            max_val = 0
            max_idx = None
            min_len_sum = 99999

            if cur_val > 0:
                prev_val = 0.0
                prev_len_sum = 0
                prev_idx = None
                if sum_t_id > 0 and src_t_id > 0:
                    prev_val = val_mat[sum_t_id - 1][src_t_id - 1]
                    prev_idx = idx_mat[sum_t_id - 1][src_t_id - 1]
                    if prev_idx:
                        prev_len_sum = prev_idx['sum_end'] - prev_idx['sum_start'] + prev_idx['src_end'] - prev_idx['src_start'] + 2
                    else:
                        prev_len_sum = 0
                max_val = prev_val + cur_val
                min_len_sum = prev_len_sum + 2
                max_idx = {'sum_start': prev_idx['sum_start'] if prev_idx else sum_t_id,
                           'sum_end': sum_t_id,
                           'src_sentid': src_sent_id,
                           'src_start': prev_idx['src_start'] if prev_idx else src_t_id,
                           'src_end': src_t_id,
                           'match_score': val_mat[sum_t_id - 1][src_t_id - 1] + cur_val,
                           'words': prev_idx['words']+[src_t] if prev_idx else [src_t],
                           'sum_wordidx': prev_idx['sum_wordidx']+[sum_t_id] if prev_idx else [sum_t_id],
                           'src_wordidx': prev_idx['src_wordidx']+[src_t_id] if prev_idx else [src_t_id]}
            else:
                if sum_t_id > 0:
                    prev_val = val_mat[sum_t_id - 1][src_t_id]
                    prev_idx = idx_mat[sum_t_id - 1][src_t_id]
                    if prev_idx:
                        prev_len_sum = prev_idx['sum_end'] - prev_idx['sum_start'] + prev_idx['src_end'] - prev_idx['src_start']
                    else:
                        prev_len_sum = 0
                    if (prev_val > max_val) or (prev_val == max_val and prev_len_sum < min_len_sum):
                        max_val = val_mat[sum_t_id - 1][src_t_id]
                        max_idx = idx_mat[sum_t_id - 1][src_t_id]
                        min_len_sum = prev_len_sum

                if src_t_id > 0:
                    prev_val = val_mat[sum_t_id][src_t_id - 1]
                    prev_idx = idx_mat[sum_t_id][src_t_id - 1]
                    if prev_idx:
                        prev_len_sum = prev_idx['sum_end'] - prev_idx['sum_start'] + prev_idx['src_end'] - prev_idx['src_start']
                    else:
                        prev_len_sum = 0
                    if (prev_val > max_val) or (prev_val == max_val and prev_len_sum < min_len_sum):
                        max_val = val_mat[sum_t_id][src_t_id - 1]
                        max_idx = idx_mat[sum_t_id][src_t_id - 1]

            val_mat[sum_t_id][src_t_id] = max_val
            idx_mat[sum_t_id][src_t_id] = max_idx

    fragment = idx_mat[len(sum_sent) - 1][len(src_sent) - 1]
    return fragment


def extract_fragments_by_LCS(sum_sent, src_sents, stemming):
    """
    LCS is not compatible with ignore_punc/ignore_stopword
    :param sum_sent:
    :param src_sents:
    :return:
    """
    match_fragments = []
    for src_sent_id, src_sent in enumerate(src_sents):
        try:
            if src_sent and len(src_sent) > 0:
                fragment = _lcs(sum_sent, src_sent, src_sent_id, stemming)
                match_fragments.append(fragment)
            else:
                match_fragments.append(None)
        except Exception:
            print("Error when doing LCS")
            print("sum_sent:" + str(sum_sent))
            print("src_sent:" + str(src_sent))
            match_fragments.append(None)
            pass

    ranked_fragments = sorted(enumerate(match_fragments), key=lambda k:k[1]['match_score'] if k[1] else 0, reverse=True)
    ranked_fragments = [t for t in ranked_fragments if t[1] is not None]

    return ranked_fragments


def extract_fragments_by_wordmatch(sum_sent, src_sents,
                                   smoothing_window=5):
    """
    Given one sum_sent, check each sent in src_sents and return one most matching fragment
    :param src_tokens:
    :param sum_tokens:
    :param smoothing_window: if no more than smoothing_window words between two fragments, merge them
    :return:
        matching_fragments: a list of triple (start_idx, end_idx, num_word) indicating the info of a fragment
        best_sent_id: id of the most matching sent
        best_fragment: triple of the most matching sent
    """
    match_fragments = []
    sum_token_set = set(sum_sent)
    for src_sent in src_sents:
        match_list = []
        # get the word match result
        for t in src_sent:
            match_flag = 1 if t in sum_token_set else 0
            match_list.append(match_flag)

        # get fragments by concatenating consecutive matching words
        fragments = []
        start = 0
        while True:
            while start < len(match_list) - 1 and match_list[start] == 0:
                start += 1
            if start >= len(match_list) - 1:
                break
            end = start
            while end < len(match_list) - 1 and match_list[end] == 1:
                end += 1
            fragments.append({'src_start': start,
                              'src_end': end - 1,
                              'match_score': end - start})
            start = end

        # smooth fragments
        if smoothing_window > 0:
            fragments = smooth_fragments(fragments, smoothing_window)

        # check if any valid fragment exists
        best_fragment = None
        best_fragment_value = 0
        for fragment in fragments:
            if fragment['match_score'] >= best_fragment_value:
                best_fragment_value = fragment['match_score']
                best_fragment = fragment

        # add sum_start and sum_end
        if best_fragment:
            sum_start, sum_end = locate_sum_fragment(sum_sent, src_sent,
                                                     best_fragment['src_start'], best_fragment['src_end'])
            best_fragment['sum_start'] = sum_start
            best_fragment['sum_end'] = sum_end

        match_fragments.append(best_fragment)

    ranked_fragments = sorted(enumerate(match_fragments), key=lambda k:k[1]['match_score'] if k[1] else 0, reverse=True)

    return ranked_fragments


def get_top_results(fragments):
    ranked_results = sorted(enumerate(fragments), key=lambda k:k[1]['match_score'] if k[1] else 0, reverse=True)
    first_sent_id, first_fragment = ranked_results[0]
    second_sent_id, second_fragment = ranked_results[1]
    return first_sent_id, first_fragment, second_sent_id, second_fragment


def match_matrix(sum_tokens, src_tokens):
    """
    :param src_tokens:
    :param sum_tokens:
    :return:
    """
    m = np.zeros((len(sum_tokens), len(src_tokens)))
    sum_idx = defaultdict(list)
    for t_id, t in enumerate(sum_tokens):
        if t.lower() == '[sep]':
            continue
        sum_idx[t].append(t_id)

    for srct_id, srct in enumerate(src_tokens):
        if srct in sum_idx:
            for sumt_id in sum_idx[srct]:
                m[sumt_id, srct_id] = 1

    return m


def shrink_matrix(matrix, src_tokens, ngram_size=2):
    """
    :param remove_shorts: due to the sparsity,
           this removes the large consecutive zero parts of matrix (along 2nd dim, src_tokens)
           also removes short n-grams which don't have a (>n+1)-grams within a window of size w
    """
    # squash it to a 1D array
    sum_matrix = np.sum(matrix, axis=0)
    fragments = []
    f_start = 0
    while True:
        # find a fragment
        if f_start >= len(sum_matrix) - 1:
            break
        while f_start < len(sum_matrix) - 1 and sum_matrix[f_start] == 0:
            f_start += 1
        f_end = f_start
        while f_end < len(sum_matrix) - 1 and sum_matrix[f_end + 1] != 0:
            f_end += 1
        # check validity of this fragment (size)
        if f_end - f_start + 1 < ngram_size:
            # print('Too short fragment (%d,%d)=%s' % (f_start, f_end, src_tokens[f_start: f_end+1]))
            f_start = f_end + 1
            continue
        fragments.append((f_start, f_end))
        # print('Find fragment (%d,%d)=%s' % (f_start, f_end, src_tokens[f_start: f_end+1]))
        f_start = f_end + 1

    # print('Find %d candidate fragments, len=%d' % (len(fragments), sum([e-s+1 for s,e in fragments])))

    # remove zero chunks
    new_m = np.array([])
    new_src = []
    ydim = matrix.shape[0]
    zero_col = np.zeros((ydim, 1))

    for f_id, (f_start, f_end) in enumerate(fragments):
        if new_m.size > 0:
            new_m = np.concatenate([new_m, zero_col, matrix[:, f_start: f_end + 1]], axis=1)
            # insert the length of segment as separator
            new_src.append('[%d]' % (f_start-fragments[f_id-1][1]))
            new_src.extend(src_tokens[f_start: f_end + 1])
        else:
            new_m = matrix[:, f_start: f_end + 1]
            new_src = src_tokens[f_start: f_end + 1]

    return new_m, new_src, fragments


def merge_fragments(src_tokens, fragments, window_size=3):

    # check if each fragment has a valid neighbour (should be used for merger)
    valid_fragments = []
    for f_id, (f_start, f_end) in enumerate(fragments):
        if f_id > 0:
            lf_start, lf_end = fragments[f_id - 1]
            if f_start - lf_end + 1 > window_size:
                continue
        if f_id < len(fragments) - 1:
            rf_start, rf_end = fragments[f_id + 1]
            if rf_start - f_end + 1 > window_size:
                continue
        valid_fragments.append((f_start, f_end))
        print('Valid fragment (%d,%d)=%s' % (f_start, f_end, src_tokens[f_start: f_end+1]))
    print('Find %d valid fragments, len=%d' % (len(valid_fragments), sum([e-s+1 for s,e in valid_fragments])))


def insert_values_to_matrix(matrix, indices, value):
    for i,j in indices:
        matrix[i, j] = value
    return matrix


def locate_fragments_in_matrix(summary_sents, source_sents, matrix, fragments):
    indices = []
    for fragment in fragments:
        sum_offset = 0
        src_offset = 0
        if fragment['sum_sentid'] > 0:
            sum_offset = fragment['sum_sentid'] + sum([len(s) for s in summary_sents[: fragment['sum_sentid']]])
        if fragment['src_sentid'] > 0:
            src_offset = fragment['src_sentid'] + sum([len(s) for s in source_sents[: fragment['src_sentid']]])

        for sum_idx in range(sum_offset + fragment['sum_start'], sum_offset + fragment['sum_end'] + 1):
            for src_idx in range(src_offset + fragment['src_start'], src_offset + fragment['src_end'] + 1):
                if matrix[sum_idx][src_idx] > 0:
                    indices.append((sum_idx, src_idx))
    return indices
