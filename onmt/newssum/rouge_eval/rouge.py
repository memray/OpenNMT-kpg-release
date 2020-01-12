# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six
from onmt.newssum.rouge_eval import rouge_score
import io
import os


class FilesRouge:
    def __init__(self, hyp_path, ref_path, metrics=None, stats=None,
                 batch_lines=None):
        assert(os.path.isfile(hyp_path))
        assert(os.path.isfile(ref_path))

        self.rouge = Rouge(metrics=metrics, stats=stats)

        def line_count(path):
            count = 0
            with open(path, "rb") as f:
                for line in f:
                    count += 1
            return count

        hyp_lc = line_count(hyp_path)
        ref_lc = line_count(ref_path)
        assert(hyp_lc == ref_lc)

        assert(batch_lines is None or type(batch_lines) == int)

        self.hyp_path = hyp_path
        self.ref_path = ref_path
        self.batch_lines = batch_lines

    def get_scores(self, avg=False, ignore_empty=False):
        """Calculate ROUGE scores between each pair of
        lines (hyp_file[i], ref_file[i]).
        Args:
          * hyp_path: hypothesis file path
          * ref_path: references file path
          * avg (False): whether to get an average scores or a list
        """
        hyp_path, ref_path = self.hyp_path, self.ref_path

        with io.open(hyp_path, encoding="utf-8", mode="r") as hyp_file:
            hyps = [line[:-1] for line in hyp_file]
        with io.open(ref_path, encoding="utf-8", mode="r") as ref_file:
            refs = [line[:-1] for line in ref_file]

        return self.rouge.get_scores(hyps, refs, avg=avg,
                                     ignore_empty=ignore_empty)


class Rouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, stopwords_removal, stemming, punctuations_removal: rouge_score.rouge_n(hyp, ref, 1, stopwords_removal, stemming, punctuations_removal),
        "rouge-2": lambda hyp, ref, stopwords_removal, stemming, punctuations_removal: rouge_score.rouge_n(hyp, ref, 2, stopwords_removal, stemming, punctuations_removal),
        "rouge-l": lambda hyp, ref, stopwords_removal, stemming, punctuations_removal: rouge_score.rouge_l_summary_level(hyp, ref, stopwords_removal, stemming, punctuations_removal),
    }
    DEFAULT_STATS = ["f", "p", "r"]
    AVAILABLE_STATS = ["f", "p", "r"]

    def __init__(self, metrics=None, stats=None, stopwords_removal=False, stemming=True, punctuations_removal=True):
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]

            for m in self.metrics:
                if m not in Rouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = Rouge.DEFAULT_METRICS

        if stats is not None:
            self.stats = [s.lower() for s in stats]

            for s in self.stats:
                if s not in Rouge.AVAILABLE_STATS:
                    raise ValueError("Unknown stat '%s'" % s)
        else:
            self.stats = Rouge.DEFAULT_STATS

        self.stopwords_removal = stopwords_removal
        self.stemming = stemming
        self.punctuations_removal = punctuations_removal

    def get_scores(self, hyps, refs, avg=False, ignore_empty=False):
        if isinstance(hyps, six.string_types):
            hyps, refs = [hyps], [refs]

        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
            hyps, refs = zip(*hyps_and_refs)

        assert(type(hyps) == type(refs))
        assert(len(hyps) == len(refs))

        if not avg:
            return self._get_scores(hyps, refs)
        return self._get_avg_scores(hyps, refs)

    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}
            hyp = [" ".join([_ for _ in hyp.split() if len(_) > 0])]
            ref = [" ".join([_ for _ in ref.split() if len(_) > 0])]
            # hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            # ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, stopwords_removal=self.stopwords_removal, stemming=self.stemming, punctuations_removal=self.punctuations_removal)
                sen_score[m] = {s: sc[s] for s in self.stats}
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}

        count = 0
        for (hyp, ref) in zip(hyps, refs):
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}
            count += 1
        scores = {m: {s: scores[m][s] / count for s in scores[m]}
                  for m in scores}
        return scores


if __name__ == '__main__':
    ref = ["originally fined # 120 for the absence from sweyne park school in essex . . unsuccessfully appealed the decision but parents still refused to pay . . grandmother paid the bailiffs and said it was a ` sober warning to others"]
    hyp = ["couple took their son out of sweyne park school in essex without permission . . they were forced to pay a # 1,230 fine after bailiffs were called in . . they told her goods would be sold to cover # 3,500 for the fine and charges . . essex county council spokesman said parents legally had to ensure attendance ."]
    r1 = rouge_score.rouge_n(hyp, ref, 1, stopwords_removal=True, stemming=False, punctuations_removal=True)
    r2 = rouge_score.rouge_n(hyp, ref, 2, stopwords_removal=True, stemming=False, punctuations_removal=True)
    rl = rouge_score.rouge_l_summary_level(hyp, ref, stopwords_removal=True, stemming=False, punctuations_removal=True)

    print(r1)
    print(r2)
    print(rl)
    """
    hyp_sent = "marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . ''"
    ref_sent = "marseille prosecutor says `` so far no videos were used in the crash investigation '' despite media reports . journalists at bild and paris match are `` very confident '' the video clip is real , an editor says . andreas lubitz had informed his lufthansa training school of an episode of severe depression , airline says ."

    stopwords_removal = False
    stemming = True
    punctuations_removal = True

    rouge = Rouge(stopwords_removal=stopwords_removal, stemming=stemming, punctuations_removal=punctuations_removal)
    score = rouge.get_scores(hyps=[hyp_sent], refs=[ref_sent])
    print(score)

    stopwords_removal = False
    stemming = True
    rouge = Rouge(stopwords_removal=stopwords_removal, stemming=stemming, punctuations_removal=punctuations_removal)
    score = rouge.get_scores(hyps=[hyp_sent], refs=[ref_sent])
    print(score)

    stopwords_removal = True
    stemming = False
    rouge = Rouge(stopwords_removal=stopwords_removal, stemming=stemming, punctuations_removal=punctuations_removal)
    score = rouge.get_scores(hyps=[hyp_sent], refs=[ref_sent])
    print(score)

    stopwords_removal = False
    stemming = False
    rouge = Rouge(stopwords_removal=stopwords_removal, stemming=stemming, punctuations_removal=punctuations_removal)
    score = rouge.get_scores(hyps=[hyp_sent], refs=[ref_sent])
    print(score)
    """


