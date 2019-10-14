import string
from os.path import join, dirname
import numpy as np
import time
import sys,logging
import matplotlib

SEP_token = "<sep>"
DIGIT_token = "<digit>"
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from nltk.stem.porter import *

stemmer = PorterStemmer()

def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]


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


def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
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
