import re
import string
from os.path import join, dirname
import numpy as np
import time
import sys,logging
import matplotlib

SEP_token = "<sep>"
DIGIT_token = "<digit>"
import time

from nltk.stem.porter import *
stemmer = PorterStemmer()

# matplotlib.use('agg')
# import matplotlib.pyplot as plt

def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]


def validate_phrases(pred_seqs, unk_token):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :return:
    '''
    valid_flags = []

    for seq in pred_seqs:
        keep_flag = True

        if len(seq) == 0:
            keep_flag = False

        if keep_flag and any([w == unk_token for w in seq]):
            keep_flag = False

        if keep_flag and any([w == '.' or w == ',' for w in seq]):
            keep_flag = False

        valid_flags.append(keep_flag)

    return np.asarray(valid_flags)


def if_present_duplicate_phrases(src_seq, tgt_seqs, stemming=True, lowercase=True):
    """
    Check if each given target sequence verbatim appears in the source sequence
    :param src_seq:
    :param tgt_seqs:
    :param stemming:
    :param lowercase:
    :param check_duplicate:
    :return:
    """
    if lowercase:
        src_seq = [w.lower() for w in src_seq]
    if stemming:
        src_seq = stem_word_list(src_seq)

    present_indices = []
    present_flags = []
    duplicate_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for tgt_seq in tgt_seqs:
        if lowercase:
            tgt_seq = [w.lower() for w in tgt_seq]
        if stemming:
            tgt_seq = stem_word_list(tgt_seq)

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_seq, tgt_seq)

        # if it reaches the end of source and no match, means it doesn't appear in the source
        present_flags.append(match_flag)
        present_indices.append(match_pos_idx)

        # check if it is duplicate
        if '_'.join(tgt_seq) in phrase_set:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
        phrase_set.add('_'.join(tgt_seq))

    assert len(present_flags) == len(present_indices)

    return np.asarray(present_flags), \
           np.asarray(present_indices), \
           np.asarray(duplicate_flags)


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break

    return match_flag, match_pos_idx


def gather_scores(gathered_scores, results_names, results_dicts):
    for result_name, result_dict in zip(results_names, results_dicts):
        for metric_name, score in result_dict.items():
            if metric_name.endswith('_num'):
                # if it's 'present_tgt_num' or 'absent_tgt_num', leave as is
                field_name = result_name
            else:
                # if it's other score like 'precision@5' is renamed to like 'present_exact_precision@'
                field_name = result_name + '_' + metric_name

            if field_name not in gathered_scores:
                gathered_scores[field_name] = []

            gathered_scores[field_name].append(score)

    return gathered_scores


def print_predeval_result(i, src_dict, tgt_seqs, present_tgt_flags,
                          pred_seqs, pred_scores, pred_idxs, copied_flags,
                          present_pred_flags, valid_pred_flags,
                          valid_and_present_flags, valid_and_absent_flags,
                          match_scores_exact, match_scores_partial,
                          results_names, results_list, score_dict):
    '''
    Print and export predictions
    '''
    # src, src_str, tgt, tgt_str_seqs, tgt_copy, pred_seq, oov
    print_out = '======================  %d =========================' % (i)
    print_out += '\n[Title]: %s \n' % (src_dict["title"])
    print_out += '[Abstract]: %s \n' % (src_dict["abstract"])
    # print_out += '[Source tokenized][%d]: %s \n' % (len(src_seq), ' '.join(src_seq))
    # print_out += 'Real Target [%d] \n\t\t%s \n' % (len(tgt_seqs), str(tgt_seqs))

    print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
                 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
    print_out += '\n'.join(
        ['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in
         zip(tgt_seqs, present_tgt_flags)])

    print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
                 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
        len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
        sum(valid_and_present_flags), sum(valid_and_absent_flags))
    print_out += ''
    preds_out = ''
    for p_id, (word, match, match_soft,
               is_valid, is_present) in enumerate(
        zip(pred_seqs, match_scores_exact, match_scores_partial,
            valid_pred_flags, present_pred_flags)):
        score = pred_scores[p_id] if pred_scores else "Score N/A"
        pred_idx = pred_idxs[p_id] if pred_idxs else "Index N/A"
        copied_flag = copied_flags[p_id] if copied_flags else "CopyFlag N/A"

        preds_out += '%s\n' % (' '.join(word))
        if is_present:
            print_phrase = '[%s]' % ' '.join(word)
        else:
            print_phrase = ' '.join(word)

        if match == 1.0:
            correct_str = '[correct!]'
        else:
            correct_str = ''

        if any(copied_flag):
            copy_str = '[copied!]'
        else:
            copy_str = ''

        pred_str = '\t\t%s\t%s \t %s %s%s\n' % ('[%.4f]' % (-score) if pred_scores else "Score N/A",
                                                print_phrase, str(pred_idx),
                                                correct_str, copy_str)
        if not is_valid:
            pred_str = '\t%s' % pred_str

        print_out += pred_str

    print_out += "\n ======================================================= \n"

    print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
                 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
    print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
                 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
        len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
        sum(valid_and_present_flags), sum(valid_and_absent_flags))

    for name, results in zip(results_names, results_list):
        # print @5@10@O@M for present_exact, print @50@M for absent_exact
        if name in ['present_exact', 'absent_exact']:
            if name.startswith('present'):
                topk_list = ['5', '10', 'k', 'M']
            else:
                topk_list = ['50', 'M']

            for topk in topk_list:
                print_out += "\n --- batch {} Corr/P/R/F1 @{}: \t".format(name, topk) \
                             + " {:6} , {:.4f} , {:.4f} , {:.4f}".format(int(results['correct@{}'.format(topk)]),
                                                                          results['precision@{}'.format(topk)],
                                                                          results['recall@{}'.format(topk)],
                                                                          results['f_score@{}'.format(topk)],
                                                                          )
                print_out += "\n --- total {} Corr/P/R/F1 @{}: \t".format(name, topk) \
                             + " {:6} , {:.4f} , {:.4f} , {:.4f}".format(
                                int(np.sum(score_dict['{}_correct@{}'.format(name, topk)])),
                                np.average(score_dict['{}_precision@{}'.format(name, topk)]),
                                np.average(score_dict['{}_recall@{}'.format(name, topk)]),
                                np.average(score_dict['{}_f_score@{}'.format(name, topk)]),)
        elif name in ['present_exact_advanced', 'absent_exact_advanced']:
            print_out += "\n --- batch {} AUC/SADR/α-nDCG@5/α-nDCG@10/nDCG/AP/MRR: \t".format(name[: name.rfind('_')]) \
                         + " {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f}".format(
                results['auc'], results['sadr'], results['alpha_ndcg@5'], results['alpha_ndcg@10'],
                results['ndcg'], results['ap'], results['mrr'],)

            print_out += "\n --- total {} AUC/SADR/α-nDCG@5/α-nDCG@10/nDCG/AP/MRR: \t".format(name[: name.rfind('_')]) \
                         + " {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f} , {:.4f}".format(
                np.average(score_dict['{}_{}'.format(name, 'auc')]),
                np.average(score_dict['{}_{}'.format(name, 'sadr')]),
                np.average(score_dict['{}_{}'.format(name, 'alpha_ndcg@5')]),
                np.average(score_dict['{}_{}'.format(name, 'alpha_ndcg@10')]),
                np.average(score_dict['{}_{}'.format(name, 'ndcg')]),
                np.average(score_dict['{}_{}'.format(name, 'ap')]),
                np.average(score_dict['{}_{}'.format(name, 'mrr')]),
            )
        else:
            # ignore partial for now
            continue

    print_out += "\n ======================================================="

    return print_out


def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits with <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens


def retain_punc_tokenize(raw_text):
    '''
    Keep almost all punctuations except ?, as ? is often caused by encoding error.
    Pad underlines before and after each punctuation.
    :param text:
    :return: a list of tokens
    '''
    puncs = string.punctuation
    pattern = r"[{}]".format(puncs)  # create the pattern

    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', raw_text)
    # pad spaces&underlines to the left and right of special punctuations
    text = re.sub(pattern, ' _\g<0>_ ', text)
    # tokenize by whitespaces
    tokens = []
    for token in re.split(r'\s', text):
        # split strings that contain letters and digits
        if re.match(r'[A-Za-z]+\d+|\d+[A-Za-z]+', token):
            token = re.findall(r'[A-Za-z]+|\d+', token)
        else:
            token = [token]
        tokens.extend(token)

    tokens = list(filter(lambda w: len(w) > 0 and w!='_?_', tokens))
    tokens = [t[1] if len(t)==3 and t[0]=='_' and t[2]=='_' else t for t in tokens]

    return tokens


def replace_numbers_to_DIGIT(tokens, k=2):
    # replace big numbers (contain more than k digit) with <digit>
    tokens = [w if not re.match('^\d{%d,}$' % k, w) else DIGIT_token for w in tokens]

    return tokens


def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % (end_ts - beg_ts))
        return retval
    return wrapper

DATA_DIR = join(dirname(dirname(__file__)), 'data')
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
MODEL_NAME = ("{:s}_model.{:s}.{:s}_contextsize.{:d}_numnoisewords.{:d}"
              "_vecdim.{:d}_batchsize.{:d}_lr.{:f}_epoch.{:d}_loss.{:f}"
              ".pth.tar")

def current_milli_time():
    return int(round(time.time() * 1000))

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def tally_parameters(model):
    if logging.getLogger() == None:
        printer = print
    else:
        printer = logging.getLogger().info

    n_params = sum([p.nelement() for p in model.parameters()])
    printer('Model name: %s' % type(model).__name__)
    printer('number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    printer('encoder: %d' % enc)
    printer('decoder: %d' % dec)

def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end='')
    sys.stdout.write(" - {:d}%".format(progress))
    sys.stdout.flush()

class Progbar(object):
    def __init__(self, logger, title, target, width=30, batch_size = None, total_examples = None, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.logger = logger
        self.title = title
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

        self.batch_size = batch_size
        self.last_batch = 0
        self.total_examples = total_examples
        self.start_time = time.time() - 0.00001
        self.last_time  = self.start_time
        self.report_delay = 10
        self.last_report  = self.start_time

    def update(self, current_epoch, current, values=[]):
        '''
        @param current: index of current step
        @param values: list of tuples (name, value_for_last_step).
        The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1

            epoch_info = '%s Epoch=%d -' % (self.title, current_epoch) if current_epoch else '%s -' % (self.title)

            barstr = epoch_info + '%%%dd/%%%dd' % (numdigits, numdigits, ) + ' (%.2f%%)['
            bar = barstr % (current, self.target, float(current)/float(self.target) * 100.0)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('.'*(prog_width-1))
                if current < self.target:
                    bar += '(-w-)'
                else:
                    bar += '(-v-)!!'
            bar += ('~' * (self.width-prog_width))
            bar += ']'
            # sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)

            # info = ''
            info = bar
            if current < self.target:
                info += ' - Run-time: %ds - ETA: %ds' % (now - self.start, eta)
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                # info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                if k == 'perplexity' or k == 'PPL':
                    info += ' - %s: %.4f' % (k, np.exp(self.sum_values[k][0] / max(1, self.sum_values[k][1])))
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))

            # update progress stats
            '''
            current_time = time.time()
            elapsed = current_time - self.last_report
            if elapsed > self.report_delay:
                trained_word_count = self.batch_size * current  # only words in vocab & sampled
                new_trained_word_count = self.batch_size * (current - self.last_batch)  # only words in vocab & sampled

                info += " - new processed %d words, %.0f words/s" % (new_trained_word_count, new_trained_word_count / elapsed)
                self.last_time   = current_time
                self.last_report = current_time
                self.last_batch  = current
            '''

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            # sys.stdout.write(info)
            # sys.stdout.flush()

            self.logger.info(info)

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                # sys.stdout.write(info + "\n")
                self.logger.critical(info + "\n")
                print(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

    def clear(self):
        self.sum_values = {}
        self.unique_values = []
        self.total_width = 0
        self.seen_so_far = 0


'''
def plot_learning_curve_and_write_csv(scores, curve_names, checkpoint_names, title, ylim=None, save_path=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    """
    train_sizes=np.linspace(1, len(scores[0]), len(scores[0]))
    plt.figure(dpi=500)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # print(train_scores)
    # print(test_scores)
    plt.grid()
    means   = {}
    stds    = {}

    # colors = "rgbcmykw"
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(curve_names)))

    for i, (name, score) in enumerate(zip(curve_names, scores)):
        # get the mean and std of score along the time step
        mean = np.asarray([np.mean(s) for s in score])
        means[name] = mean
        std  = np.asarray([np.std(s) for s in score])
        stds[name] = std

        if name.lower().startswith('training ml'):
            score_ = [np.asarray(s) / 20.0 for s in score]
            mean = np.asarray([np.mean(s) for s in score_])
            std  = np.asarray([np.std(s) for s in score_])

        plt.fill_between(train_sizes, mean - std,
                         mean + std, alpha=0.1,
                         color=colors[i])
        plt.plot(train_sizes, mean, 'o-', color=colors[i],
                 label=name)

    plt.legend(loc="best", prop={'size': 6})
    # plt.show()
    if save_path:
        plt.savefig(save_path + '.png', bbox_inches='tight')

        csv_lines = ['time, ' + ','.join(curve_names)]
        for t_id, time in enumerate(checkpoint_names):
            csv_line = time + ',' + ','.join([str(means[c_name][t_id]) for c_name in curve_names])
            csv_lines.append(csv_line)

        with open(save_path + '.csv', 'w') as result_csv:
            result_csv.write('\n'.join(csv_lines))

    plt.close()
    return plt
'''
