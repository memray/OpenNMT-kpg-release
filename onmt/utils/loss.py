"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
        )
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
            lambda_coverage=opt.lambda_coverage,
            lambda_orth_reg=opt.lambda_orth_reg,
            lambda_sem_cov=opt.lambda_sem_cov
        )
    else:
        compute = NMTLossCompute(
            criterion, loss_gen,
            lambda_coverage=opt.lambda_coverage,
            lambda_orth_reg=opt.lambda_orth_reg,
            lambda_sem_cov=opt.lambda_sem_cov)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class ReplayMemory(object):

    def __init__(self, capacity=300):
        # vanilla replay memory
        self.capacity = capacity
        self.memory = []

    def push(self, stuff):
        """Saves a transition."""
        self.memory.append(stuff)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_orth_reg=0.0, lambda_sem_cov=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.lambda_orth_reg = lambda_orth_reg
        self.lambda_sem_cov = lambda_sem_cov
        self.replay_buffer = ReplayMemory(300)

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            "states": attns.get("states")
        }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        return shard_state

    def _compute_loss(self, batch, output, target,
                      std_attn=None, coverage_attn=None,
                      states=None):

        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss

        # compute orthogonal penalty loss
        target_indices = target
        decoder_hidden_states = states
        target_sep_idx = batch.sep_indices
        if self.lambda_orth_reg != 0.0:
            # decoder hidden state: output of decoder
            orthogonal_penalty = self._compute_orthogonal_regularization_loss(target_indices, decoder_hidden_states, target_sep_idx)
            loss += orthogonal_penalty
        if self.lambda_sem_cov != 0.0:
            # model: model, has to include
            #   target_encoding_mlp: an mlp with parameter (target_encoder_dim, target_encoding_mlp_hidden_dim), with non-linearity function
            #   bilinear_layer: nn.Bilinear(source_hid, target_encoding_mlp_hidden_dim, 1), without non-linearity function
            # source_representations: batch x source_len x source_hid
            # target_representations: output of target encoder (last state), batch x target_hid
            semantic_coverage_loss = self._compute_semantic_coverage_loss(model, source_representations, target_representations, target_indices, sep_idx, n_neg)
            loss += semantic_coverage_loss

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum(2).view(-1)
        covloss *= self.lambda_coverage
        return covloss

    def orthogonal_penalty(self, _m, I, l_n_norm=2):
        # _m: h x n
        # I:  n x n
        m = torch.mm(torch.t(_m), _m)  # n x n
        return torch.norm((m - I), p=l_n_norm)

    def _compute_orthogonal_regularization_loss(self, target_indices, decoder_hidden_states, sep_idx):
        # target_indices: batch x target_len
        # decoder_hidden_states: batch x target_len x hid
        # aux loss: make the decoder outputs at all <sep>s to be orthogonal
        penalties = []
        for i in range(len(target_indices)):
            # per data point in a batch
            seps = []
            for j in range(len(target_indices[i])):  # len of target
                if target_indices[i][j] == sep_idx:
                    seps.append(decoder_hidden_states[i][j])
            if len(seps) > 1:
                seps = torch.stack(seps, -1)  # hid x n
                identity = torch.eye(seps.size(-1))  # n x n
                if decoder_hidden_states.is_cuda:
                    identity = identity.cuda()
                penalty = self.orthogonal_penalty(seps, identity, 2)  # 1
                penalties.append(penalty)

        if len(penalties) > 0 and decoder_hidden_states.size(0) > 0:
            penalties = torch.sum(torch.stack(penalties, -1)) / float(decoder_outputs.size(0))
        else:
            penalties = 0.0
        penalties = penalties * self.lambda_orth_reg
        return penalties

    def random_insert(self, _list, elem):
        insert_before_this = np.random.randint(low=0, high=len(_list) + 1)
        return _list[:insert_before_this] + [elem] + _list[insert_before_this:], insert_before_this

    def _compute_semantic_coverage_loss(self, model, source_representations, target_representations, target_indices, sep_idx, n_neg):
        # source_representations: batch x source_hid
        # target_representations: batch x target_len x target_hid
        # target_indices: batch x target_len
        batch_size = target_representations.size(0)
        # n_neg is how many negative samples to sample
        batch_inputs_source, batch_inputs_target, batch_labels = [], [], []
        source_representations = source_representations.detach()
        for b in range(batch_size):
            # 0. find sep positions
            for i in range(len(target_indices[b])):  # target len
                if target_indices[b][i] == sep_idx:
                    trg_rep = target_representations[b][i]  # hid
                    # 1. negative sampling
                    if len(self.replay_memory) >= n_neg:
                        neg_list = self.replay_memory.sample(n_neg)
                        inputs, which = self.random_insert(neg_list, source_representations[b])
                        inputs = torch.stack(inputs, 0)  # n_neg+1 x hid
                        batch_inputs_source.append(inputs)
                        batch_inputs_target.append(trg_rep)
                        batch_labels.append(which)
            # 2. push source representations into replay memory
            self.replay_memory.push(source_representations[b])
        if len(batch_inputs_source) == 0:
            return 0.0
        batch_inputs_source = torch.stack(batch_inputs_source, 0)  # big_batch x n_neg+1 x source_hid
        batch_inputs_target = torch.stack(batch_inputs_target, 0)  # big_batch x target_hid
        batch_labels = np.array(batch_labels)  # big_batch
        batch_labels = torch.autograd.Variable(torch.from_numpy(batch_labels.copy()).type(torch.LongTensor))
        if target_representations.is_cuda:
            batch_labels = batch_labels.cuda()
        # 3. prediction
        batch_inputs_target = model.target_encoding_mlp(batch_inputs_target)  # big_batch x mlp_hid
        batch_inputs_target = torch.stack([batch_inputs_target] * batch_inputs_source.size(1), 1)  # big_batch x n_neg+1 x mlp_hid
        pred = model.bilinear_layer(batch_inputs_source, batch_inputs_target).squeeze(-1)  # batch x n_neg+1
        pred = torch.nn.functional.log_softmax(pred, dim=-1)  # batch x n_neg+1

        # 4. loss compute
        loss = self.criterion(pred, batch_labels)
        loss = loss * self.lambda_sem_cov
        return loss

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
