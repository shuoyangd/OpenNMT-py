import torch
import torch.nn as nn
from onmt.Loss import LossComputeBase

class RNNGLoss(LossComputeBase):
  
  def __init__(self, generator, tgt_vocab, lambda_=1.0):
    super(RNNGLoss, self).__init__(generator, tgt_vocab)

    weight = torch.ones(len(tgt_vocab))
    weight[self.padding_idx] = 0
    self.criterion = nn.NLLLoss(weight, size_average=False)
    self.lambda_ = lambda_

  def make_shard_state(self, batch, output, range_, attns=None):
    """ See base class for args description. """
    return {
        "tokens": batch.tgt[range_[0] + 1: range_[1]],
        "output": output,
        "target": batch.aux_tgt[range_[0] + 1: range_[1]],
    }

  def compute_loss(self, batch, tokens, output, target):
    scores = self.generator(tokens, output, actions=target)
    scores = scores.view(-1, scores.size(2))
    scores_data = scores.data.clone()

    target = target.view(-1)
    target_data = target.data.clone()

    loss = self.criterion(scores, target) * self.lambda_
    loss_data = loss.data.clone()

    stats = self.stats(loss_data, scores_data, target_data)

    return loss, stats

