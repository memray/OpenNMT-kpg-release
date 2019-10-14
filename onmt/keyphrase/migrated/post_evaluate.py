import json
import logging
from nltk.stem.porter import *
import numpy as np

import os
import sys

from pykp import io
from pykp.io import load_json_data


def check_if_present(source_tokens, targets_tokens):
    target_present_flags = []
    for target_tokens in targets_tokens:
        # whether do filtering on groundtruth phrases.
        present = False
        for i in range(len(source_tokens) - len(target_tokens) + 1):
            match = None
            for j in range(len(target_tokens)):
                if target_tokens[j] != source_tokens[i + j]:
                    match = False
                    break
            if len(target_tokens) > 0 and j == len(target_tokens) - 1 and match == None:
                present = True
                break

        target_present_flags.append(present)
    assert len(target_present_flags) == len(targets_tokens)

    return target_present_flags

def get_match_flags(targets, predictions):
    match_flags = np.asarray([0] * len(predictions), dtype='int32')
    for pid, predict in enumerate(predictions):
        for stemmed_target in targets:
            if len(stemmed_target) == len(predict):
                match_flag = True
                for i, w in enumerate(predict):
                    if predict[i] != stemmed_target[i]:
                        match_flag = False
                if match_flag:
                    match_flags[pid] = 1
                    break
    return match_flags

def evaluate_(source_str_list, targets_str_list, prediction_str_list,
              model_name, dataset_name,
              filter_criteria='present',
              matching_after_stemming=True,
              output_path=None):
    '''
    '''
    assert filter_criteria in ['absent', 'present', 'all']
    stemmer = PorterStemmer()

    if output_path != None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, model_name)):
            os.makedirs(os.path.join(output_path, model_name))

    json_writer = open(os.path.join(output_path, model_name, '%s.json' % dataset_name), 'w+')
    score_csv_path = os.path.join(output_path, 'all_scores.csv')
    csv_writer = open(score_csv_path, 'a')

    print('Evaluating on %s@%s' % (model_name, dataset_name))
    # Evaluation part
    macro_metrics = []
    macro_matches = []

    total_source_length = 0
    length_groundtruth = []
    length_groundtruth_for_evaluate = []
    number_groundtruth = []
    number_groundtruth_for_evaluate = []
    total_number_groundtruth = 0
    total_number_groundtruth_for_evaluate = 0
    total_groundtruth_set = set()
    total_groundtruth_set_for_evaluate = set()

    # remove the empty targets first
    new_targets_str_list = []
    for targets_str in targets_str_list:
        new_targets_str = []
        for target_str in targets_str:
            if len(target_str.strip()) > 0:
                new_targets_str.append(target_str.strip())
        new_targets_str_list.append(new_targets_str)

    targets_str_list = new_targets_str_list

    real_test_size = 0

    """
    Iterate each document
    """
    for doc_id, (source_text, targets, predictions)\
            in enumerate(zip(source_str_list, targets_str_list, prediction_str_list)):
        # print(targets)
        # print(predictions)
        # print('*' * 100)

        # if doc_id > 5:
        #     break
        if doc_id + 1 % 1000 == 0:
            print(doc_id)

        '''
        stem all texts/targets/predictions
        '''
        stemmed_source_text_tokens = [stemmer.stem(t).strip().lower() for t in io.copyseq_tokenize(source_text)]
        stemmed_targets_tokens = [[stemmer.stem(w).strip().lower() for w in io.copyseq_tokenize(target)] for target in targets]
        stemmed_predictions_tokens = [[stemmer.stem(w).strip().lower() for w in io.copyseq_tokenize(prediction)] for prediction in predictions]

        '''
        check and filter targets/predictions by whether it appear in source text
        '''
        if filter_criteria != 'all':
            if matching_after_stemming:
                source_tokens_to_match = stemmed_source_text_tokens
                targets_tokens_to_match = stemmed_targets_tokens
                predictions_tokens_to_match = stemmed_predictions_tokens
            else:
                source_tokens_to_match = io.copyseq_tokenize(source_text.strip().lower())
                targets_tokens_to_match = [io.copyseq_tokenize(target.strip().lower()) for target in targets]
                predictions_tokens_to_match = [io.copyseq_tokenize(prediction.strip().lower()) for prediction in predictions]

            target_present_flags = check_if_present(source_tokens_to_match, targets_tokens_to_match)
            prediction_present_flags = check_if_present(source_tokens_to_match, predictions_tokens_to_match)

            if filter_criteria == 'present':
                targets_valid_flags = target_present_flags
                prediction_valid_flags = prediction_present_flags
            elif filter_criteria == 'absent':
                targets_valid_flags = [not f for f in target_present_flags]
                prediction_valid_flags = [not f for f in prediction_present_flags]

            targets_for_evaluate = np.asarray(targets)[targets_valid_flags].tolist()
            stemmed_targets_for_evaluate = np.asarray(stemmed_targets_tokens)[targets_valid_flags].tolist()
            predictions_for_evaluate = np.asarray(predictions)[prediction_valid_flags].tolist()
            stemmed_predictions_for_evaluate = np.asarray(stemmed_predictions_tokens)[prediction_valid_flags].tolist()

        else:
            targets_for_evaluate = targets
            stemmed_targets_for_evaluate = stemmed_targets_tokens
            predictions_for_evaluate = predictions
            stemmed_predictions_for_evaluate = stemmed_predictions_tokens

        total_source_length += len(source_tokens_to_match)
        total_number_groundtruth += len(targets)
        total_number_groundtruth_for_evaluate += len(targets_for_evaluate)

        number_groundtruth.append(len(targets))
        number_groundtruth_for_evaluate.append(len(targets_for_evaluate))

        for target in targets:
            total_groundtruth_set.add(' '.join(target))
            length_groundtruth.append(len(target))
        for target in targets_for_evaluate:
            total_groundtruth_set_for_evaluate.add(' '.join(target))
            length_groundtruth_for_evaluate.append(len(target))

        if len(targets_for_evaluate) > 0:
            real_test_size += 1

        # """
        '''
        check each prediction if it can match any ground-truth target
        '''
        valid_predictions_match_flags = get_match_flags(stemmed_targets_for_evaluate, stemmed_predictions_for_evaluate)
        predictions_match_flags = get_match_flags(stemmed_targets_for_evaluate, stemmed_predictions_tokens)
        '''
        Compute metrics
        '''
        metric_dict = {}
        for number_to_predict in [5, 10]:
            metric_dict['target_number'] = len(targets_for_evaluate)
            metric_dict['prediction_number'] = len(predictions_for_evaluate)
            metric_dict['correct_number@%d' % number_to_predict] = sum(valid_predictions_match_flags[:number_to_predict])

            # Precision
            metric_dict['p@%d' % number_to_predict] = float(sum(valid_predictions_match_flags[:number_to_predict])) / float(
                number_to_predict)

            # Recall
            if len(targets_for_evaluate) != 0:
                metric_dict['r@%d' % number_to_predict] = float(sum(valid_predictions_match_flags[:number_to_predict])) \
                                                          / float(len(targets_for_evaluate))
            else:
                metric_dict['r@%d' % number_to_predict] = 0

            # F-score
            if metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict] != 0:
                metric_dict['f1@%d' % number_to_predict] = 2 * metric_dict['p@%d' % number_to_predict] * metric_dict[
                    'r@%d' % number_to_predict] / float(
                    metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict])
            else:
                metric_dict['f1@%d' % number_to_predict] = 0

            # Bpref: binary preference measure
            bpref = 0.
            trunked_match = valid_predictions_match_flags[:number_to_predict].tolist()  # get the first K prediction to evaluate
            match_indexes = np.nonzero(trunked_match)[0]

            if len(match_indexes) > 0:
                for mid, mindex in enumerate(match_indexes):
                    bpref += 1. - float(mindex - mid) / float(
                        number_to_predict)  # there're mindex elements, and mid elements are correct, before the (mindex+1)-th element
                metric_dict['bpref@%d' % number_to_predict] = float(bpref) / float(len(match_indexes))
            else:
                metric_dict['bpref@%d' % number_to_predict] = 0

            # MRR: mean reciprocal rank
            rank_first = 0
            try:
                rank_first = trunked_match.index(1) + 1
            except ValueError:
                pass

            if rank_first > 0:
                metric_dict['mrr@%d' % number_to_predict] = float(1) / float(rank_first)
            else:
                metric_dict['mrr@%d' % number_to_predict] = 0

        macro_metrics.append(metric_dict)
        macro_matches.append(valid_predictions_match_flags)

        '''
        Print information on each prediction
        '''
        print_out = '[DOC_ID] %d\n' % doc_id
        print_out += '[SOURCE][{0}]: {1}\n'.format(len(source_text) , source_text)
        print_out += '[STEMMED SOURCE][{0}]: {1}'.format(len(stemmed_source_text_tokens) , ' '.join(stemmed_source_text_tokens))
        print_out += '\n'

        print_out += '[TARGET]: %d/%d valid/all targets\n' % (len(targets_for_evaluate), len(targets))
        for target, stemmed_target, targets_valid_flag in zip(targets, stemmed_targets_tokens, targets_valid_flags):
            if targets_valid_flag:
                print_out += '\t\t%s (%s)\n' % (target, ' '.join(stemmed_target))
        for target, stemmed_target, targets_valid_flag in zip(targets, stemmed_targets_tokens, targets_valid_flags):
            if not targets_valid_flag:
                print_out += '\t\t[ABSENT]%s (%s)\n' % (target, ' '.join(stemmed_target))

        print_out += '\n'

        num_correct_5 = sum(predictions_match_flags[:5]) if len(predictions_match_flags) >=5 else sum(predictions_match_flags)
        num_correct_10 = sum(predictions_match_flags[:10]) if len(predictions_match_flags) >=10 else sum(predictions_match_flags)
        print_out += '[DECODE]: %d/%d valid/all predictions, #(correct@5)=%d, #(correct@10)=%d' \
            % (len(predictions_for_evaluate), len(predictions), num_correct_5, num_correct_10)
        for prediction, stemmed_prediction, prediction_present_flag, predictions_match_flag \
                in zip(predictions, stemmed_predictions_tokens, prediction_present_flags, predictions_match_flags):
            if prediction_present_flag:
                print_out += ('\n\t\t%s (%s)' % (prediction, ' '.join(stemmed_prediction)))
            else:
                print_out += ('\n\t\t[ABSENT]%s (%s)' % (prediction, ' '.join(stemmed_prediction)))
            if predictions_match_flag == 1:
                print_out += ' [correct!]'
        # c += '\n'
        # for prediction, stemmed_prediction, prediction_present_flag, predictions_match_flag \
        #         in zip(predictions, stemmed_predictions_tokens, prediction_present_flags, predictions_match_flags):
        #     if not prediction_present_flag:
        #         c += ('\n\t\t[ABSENT]%s (%s)' % (prediction, ' '.join(stemmed_prediction)))
        #         if predictions_match_flag == 1:
        #             c += ' [correct!]'


        # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
        # if inputs_unk is not None:
        #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
        #     logger.info(k)
        # a += k

        for number_to_predict in [5, 10]:
            print_out += '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f, Bpref=%.4f, MRR=%.4f' % (
            number_to_predict, metric_dict['p@%d' % number_to_predict], metric_dict['r@%d' % number_to_predict],
            metric_dict['f1@%d' % number_to_predict], metric_dict['bpref@%d' % number_to_predict], metric_dict['mrr@%d' % number_to_predict])

        # logger.info(print_out)
        # logger.info('*' * 100)

        out_dict = {}
        out_dict['src_str'] = source_text
        out_dict['trg_str'] = targets
        out_dict['trg_present_flag'] = target_present_flags
        out_dict['pred_str'] = predictions
        out_dict['pred_score'] = [0.0] * len(predictions)
        out_dict['present_flag'] = prediction_present_flags
        out_dict['valid_flag'] = [True] * len(predictions)
        out_dict['match_flag'] = [float(m) for m in predictions_match_flags]

        # print(out_dict)

        json_writer.write(json.dumps(out_dict)+'\n')

        assert len(out_dict['trg_str']) == len(out_dict['trg_present_flag'])
        assert len(out_dict['pred_str']) == len(out_dict['present_flag']) \
               == len(out_dict['valid_flag']) == len(out_dict['match_flag']) == len(out_dict['pred_score'])
        # """

    logger.info('Avg(Source Text Length)=%.4f' % (float(total_source_length) / len(source_str_list)))
    logger.info('#(Target)=%d' % (len(length_groundtruth)))
    logger.info('Avg(Target Length)=%.4f' % (np.mean(length_groundtruth)))
    logger.info('#(%s Target)=%d' % (filter_criteria.upper(), len(length_groundtruth_for_evaluate)))
    logger.info('Avg(%s Target Length)=%.4f' % (filter_criteria.upper(), np.mean(length_groundtruth_for_evaluate)))

    logger.info('#(Ground-truth Keyphrase)=%d' % total_number_groundtruth)
    logger.info('#(%s Ground-truth Keyphrase)=%d' % (filter_criteria.upper(), total_number_groundtruth_for_evaluate))
    logger.info('Avg(Ground-truth Keyphrase)=%.4f' % (float(total_number_groundtruth) / len(source_str_list)))
    logger.info('Avg(%s Ground-truth Keyphrase)=%.4f' % (filter_criteria.upper(),
                                                     float(total_number_groundtruth_for_evaluate) / len(source_str_list)))

    logger.info('#(Unique Ground-truth Keyphrase)=%d' % (len(total_groundtruth_set)))
    logger.info('#(Unique %s Ground-truth Keyphrase)=%d' % (filter_criteria.upper(), len(total_groundtruth_set_for_evaluate)))

    logger.info('Avg(Ground-truth Keyphrase)=%.4f' % (np.mean(number_groundtruth)))
    logger.info('Var(Ground-truth Keyphrase)=%.4f' % (np.var(number_groundtruth)))
    logger.info('Std(Ground-truth Keyphrase)=%.4f' % (np.std(number_groundtruth)))

    logger.info('Avg(%s Ground-truth Keyphrase)=%.4f' % (filter_criteria.upper(), np.mean(number_groundtruth_for_evaluate)))
    logger.info('Var(%s Ground-truth Keyphrase)=%.4f' % (filter_criteria.upper(), np.var(number_groundtruth_for_evaluate)))
    logger.info('Std(%s Ground-truth Keyphrase)=%.4f' % (filter_criteria.upper(), np.std(number_groundtruth_for_evaluate)))

    '''
    Export the f@5 and f@10 for significance test
    '''
    # for k in [5, 10]:
    #     with open(config['predict_path'] + '/macro-f@%d-' % (k) + model_name+'-'+dataset_name+'.txt', 'w') as writer:
    #         writer.write('\n'.join([str(m['f1@%d' % k]) for m in macro_metrics]))

    # """
    '''
    Compute the corpus evaluation
    '''
    overall_score = {}

    for k in [5, 10]:
        correct_number = sum([m['correct_number@%d' % k] for m in macro_metrics])
        overall_target_number = sum([m['target_number'] for m in macro_metrics])
        overall_prediction_number = sum([m['prediction_number'] for m in macro_metrics])

        if real_test_size * k < overall_prediction_number:
            overall_prediction_number = real_test_size * k

        overall_score['target_number'] = sum([m['target_number'] for m in macro_metrics])
        overall_score['correct_number@%d' % k] = sum([m['correct_number@%d' % k] for m in macro_metrics])
        overall_score['prediction_number@%d' % k] = overall_prediction_number

        # Compute the macro Measures, by averaging the macro-score of each prediction
        overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in macro_metrics])) / float(real_test_size)

        # Print basic statistics
        logger.info('%s@%s' % (model_name, dataset_name))
        output_str = 'Overall - valid testing data=%d, Number of Target=%d/%d, ' \
                     'Number of Prediction=%d, Number of Correct=%d' % (
                    real_test_size,
                    overall_target_number, total_number_groundtruth,
                    overall_prediction_number, correct_number
        )
        logger.info(output_str)

        # Print macro-average performance
        output_str = 'macro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['p@%d' % k],
                    k, overall_score['r@%d' % k],
                    k, overall_score['f1@%d' % k]
        )
        logger.info(output_str)

        # Compute the binary preference measure (Bpref)
        overall_score['bpref@%d' % k] = float(sum([m['bpref@%d' % k] for m in macro_metrics])) / float(real_test_size)

        # Compute the mean reciprocal rank (MRR)
        overall_score['mrr@%d' % k] = float(sum([m['mrr@%d' % k] for m in macro_metrics])) / float(real_test_size)

        output_str = '\t\t\tBpref@%d=%f, MRR@%d=%f' % (
                    k, overall_score['bpref@%d' % k],
                    k, overall_score['mrr@%d' % k]
        )
        logger.info(output_str)

    csv_writer.write('%s, %s, '
                     '%d, %d, %d, %d, %d, %d, '
                     '%f, %f, %f, %f, %f, '
                     '%f, %f, %f, %f, %f\n' % (
                model_name, dataset_name,
                len(source_str_list), real_test_size,
                total_number_groundtruth, total_number_groundtruth_for_evaluate,
                overall_score['correct_number@%d' % 5], overall_score['correct_number@%d' % 10],

                overall_score['p@%d' % 5],
                overall_score['r@%d' % 5],
                overall_score['f1@%d' % 5],
                overall_score['bpref@%d' % 5],
                overall_score['mrr@%d' % 5],

                overall_score['p@%d' % 10],
                overall_score['r@%d' % 10],
                overall_score['f1@%d' % 10],
                overall_score['bpref@%d' % 10],
                overall_score['mrr@%d' % 10]
    ))

    json_writer.close()
    csv_writer.close()
    # """

def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    # ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)
    # logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging


def load_predictions_from_file(prediction_dir, file_suffix='.txt'):
    predictions_str_dict = {}

    for pred_file_name in os.listdir(prediction_dir):
        if not pred_file_name.endswith(file_suffix):
            continue
        doc_id = pred_file_name[: pred_file_name.find(file_suffix)]
        prediction_str_list = []
        with open(os.path.join(prediction_dir, pred_file_name), 'r') as pred_file:
            for line in pred_file:
                prediction_str_list.append(line.strip())
        predictions_str_dict[doc_id] = prediction_str_list
    sorted_predictions_str_dict = sorted(predictions_str_dict.items(), key=lambda k:k[0])
    doc_ids = [d[0] for d in sorted_predictions_str_dict]
    predictions_str_list = [d[1] for d in sorted_predictions_str_dict]
    # print(doc_ids)
    return predictions_str_list


def load_plain_text(text_dir):
    source_text_dict = {}

    for source_file_name in os.listdir(text_dir):
        if not source_file_name.endswith('.txt'):
            continue
        doc_id = source_file_name[: source_file_name.find('.txt')]
        with open(os.path.join(text_dir, source_file_name), 'r') as pred_file:
            text = ' '.join([l.strip() for l in pred_file.readlines()])
            # postag = [t.split('_')[1] for t in text_tokens]
            source_text_dict[doc_id] = text

    sorted_source_text_dict = sorted(source_text_dict.items(), key=lambda k:k[0])
    doc_ids = [d[0] for d in sorted_source_text_dict]
    source_text_list = [d[1] for d in sorted_source_text_dict]
    # print(doc_ids)

    return source_text_list


def load_postag_text(postag_text_dir):
    source_text_dict = {}

    for source_file_name in os.listdir(postag_text_dir):
        if not source_file_name.endswith('.txt'):
            continue
        doc_id = source_file_name[: source_file_name.find('.txt')]
        with open(os.path.join(postag_text_dir, source_file_name), 'r') as pred_file:
            text_tokens = (' '.join(pred_file.readlines())).split()
            text = ' '.join([t.split('_')[0] for t in text_tokens])
            # postag = [t.split('_')[1] for t in text_tokens]
            source_text_dict[doc_id] = text

    sorted_source_text_dict = sorted(source_text_dict.items(), key=lambda k:k[0])
    doc_ids = [d[0] for d in sorted_source_text_dict]
    source_text_list = [d[1] for d in sorted_source_text_dict]
    # print(doc_ids)

    return source_text_list


def evaluate_baselines(models, test_sets, output_dir, filter_criteria, plain_or_postag):
    '''
    evaluate baselines' performance
    plain_or_postag: specify the source of source text. The postag text leads to more bias, but this is the way used in Meng 17.
    :return:
    '''
    base_dir = 'prediction/'
    print(os.path.abspath(base_dir))

    if not os.path.exists(os.path.join(base_dir, output_dir)):
        os.makedirs(os.path.join(base_dir, output_dir))
    score_csv_path = os.path.join(base_dir, output_dir, 'all_scores.csv')
    with open(score_csv_path, 'w+') as csv_writer:
        csv_writer.write('model, data, '
                         '#doc, #valid_doc, #tgt, #valid_tgt, #corr@5, #corr@10, '
                         'p@5, r@5, f1@5, bpref@5, mrr@5, '
                         'p@10, r@10, f1@10, bpref@10, mrr@10\n')

    for model_name in models:
        for dataset_name in test_sets:

            if dataset_name == 'stackexchange':
                src_fields = ['title', 'question']
                trg_fields = ['tags']
                valid_check = True
            elif dataset_name == 'twacg':
                src_fields = ['observation']
                trg_fields = ['admissible_commands']
            else:
                src_fields = ['title', 'abstract']
                trg_fields = ['keyword']

            if plain_or_postag == 'postag':
                postag_text_dir = '/Users/memray/Project/keyphrase/seq2seq-keyphrase/dataset/keyphrase/baseline-data/%s/text' % dataset_name
                source_str_list = load_postag_text(postag_text_dir)
            elif plain_or_postag == 'plain':
                plain_text_dir = '/Users/memray/Project/keyphrase/seq2seq-keyphrase/dataset/keyphrase/baseline-data/%s/plain_text' % dataset_name
                source_str_list = load_plain_text(plain_text_dir)
            else:
                raise NotImplementedError

            targets_dir = '/Users/memray/Project/keyphrase/seq2seq-keyphrase/dataset/keyphrase/baseline-data/%s/keyphrase' % dataset_name
            targets_str_list = load_predictions_from_file(targets_dir, file_suffix='.txt')
            prediction_dir = os.path.join(base_dir, model_name, dataset_name)

            if not os.path.exists(prediction_dir):
                print('Folder not found: %s' % prediction_dir)
                continue

            prediction_str_list = load_predictions_from_file(prediction_dir, file_suffix='.txt.phrases')
            # prediction_str_list = [[] for i in range(len(targets_str_list))]
            print(dataset_name)
            print('#(src)=%d' % len(source_str_list))
            print('#(tgt)=%d' % len(targets_str_list))
            print('#(preds)=%d' % len(prediction_str_list))
            evaluate_(source_str_list, targets_str_list, prediction_str_list, model_name, dataset_name, filter_criteria,
                      matching_after_stemming = True,
                      output_path=os.path.join(base_dir, output_dir))

            #if model_name == 'Maui':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/maui_output/' + dataset_name
            #if model_name == 'Kea':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/kea_output/' + dataset_name


"""
def significance_test():
    model1 = 'CopyRNN'
    models = ['TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'RNN', 'CopyRNN']

    test_sets = config['testing_datasets']

    def load_result(filepath):
        with open(filepath, 'r') as reader:
            return [float(l.strip()) for l in reader.readlines()]

    for model2 in models:
        print('*'*20 + '  %s Vs. %s  ' % (model1, model2) + '*' * 20)
        for dataset_name in test_sets:
            for k in [5, 10]:
                print('Evaluating on %s@%d' % (dataset_name, k))
                filepath = config['predict_path'] + '/macro-f@%d-' % (k) + model1 + '-' + dataset_name + '.txt'
                val1 = load_result(filepath)
                filepath = config['predict_path'] + '/macro-f@%d-' % (k) + model2 + '-' + dataset_name + '.txt'
                val2 = load_result(filepath)
                s_test = scipy.stats.wilcoxon(val1, val2)
                print(s_test)
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
print('Log path: %s' % (os.path.abspath('prediction/post_evaluate.log')))
logger = init_logging(os.path.abspath('prediction/post_evaluate.log'))

if __name__ == '__main__':
    # 'TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'KEA', 'RNN_present', 'CopyRNN_present_singleword=0', 'CopyRNN_present_singleword=1', 'CopyRNN_present_singleword=2'
    filter_criteria = 'present' # we don't have absent predictions for these models actually
    # models = ['TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'KEA', 'CopyRNN_meng17']
    models = ['CopyRNN_meng17']

    test_sets = ['duc', 'kp20k']
    # test_sets = ['duc', 'inspec', 'nus', 'semeval', 'krapivin', 'kp20k']
    # test_sets = ['stackexchange']
    # test_sets = ['twacg']

    # plain is only available for ['duc', 'kp20k']
    plain_or_postag = 'plain'
    # plain_or_postag = 'postag'

    evaluate_baselines(models, test_sets, output_dir='output_json_%s_20190415' % plain_or_postag,
                       filter_criteria=filter_criteria, plain_or_postag=plain_or_postag)
    # significance_test()
