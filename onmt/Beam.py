from __future__ import division
import torch
import onmt
import pdb
from scipy.stats import multivariate_normal
#vocab.stoi[onmt.IO.BOS_WORD]
#self.vocab.stoi[onmt.IO.EOS_WORD]
class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos, src_length,
                 n_best=1, cuda=False, vocab=None,
                 global_scorer=None, min_length = 0,
                 length_prior = None, 
                 length_prior_factor = None, 
                 length_penalty = None):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                       .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        #length penalty stuff
        self.min_length = min_length
        self.length_prior = length_prior
        self.length_prior_factor = length_prior_factor if self.length_prior is not None else 0.0
        self.length_penalty = length_penalty if length_penalty is not None else 0.0
        self.src_length = src_length
        print('created beam', self.src_length, self.min_length)

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1) #target size vocab size
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)
        #best_scores dim (beam_size) FloatTensor
        #best_scores_id dim (beam_size) LongTensor

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words # some fancy indexing is happening here... best_scores_id has ids > than vocab size
        self.prev_ks.append(prev_k)
        tmp = (best_scores_id - prev_k * num_words)
        #print('Ids', self.vocab.itos[tmp[0]])
        #print('best_scores', best_scores)
        #print(len(self.next_ys))
        #self.next_ys.append((best_scores_id - prev_k * num_words))
        self.next_ys.append(tmp)
        self.attn.append(attn_out.index_select(0, prev_k))

        if self.global_scorer is not None:
            self.global_scorer.updateGlobalState(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                if self.length_prior is not None:
                    lp = multivariate_normal.logpdf([self.src_length, len(self.next_ys) - 1], 
                                                        mean = self.length_prior[0], 
                                                        cov = self.length_prior[1])
                else:
                    lp = 0.0
                #print('added to finished', s, self.src_length,  len(self.next_ys) - 1, i, lp, s + self.length_prior_factor *lp)
                #pdb.set_trace()
                finished_score = s + (len(self.next_ys) * self.length_penalty)
                finished_score =  ((1.0 - self.length_prior_factor) * finished_score) + \
                                  (self.length_prior_factor * lp)

                self.finished.append((finished_score, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos: 
            # self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def updateGlobalState(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])
