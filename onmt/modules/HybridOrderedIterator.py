import sys
import codecs
import argparse
import lazy_io
import math
import random
import torch
from torch.autograd import Variable

class HybridOrderedIterator:
    def __init__(self, train, batch_size, utts_file, feats_file, vocab_file):
      self.reader = lazy_io.read_dict_scp(feats_file)
      self.utts_file = utts_file
      self.feats_file = feats_file
      self.tgt_vocab = torch.load(vocab_file)[1][1]
      self.train = train
      self.batch_size = batch_size


    def __len__(self):
      return math.ceil(3696 / self.batch_size)


    def __iter__(self):
      self.create_batches()
      for batch in self.batches:
        yield batch


    def data(self):
       idx = 0
       with open(self.utts_file, "r") as f:
          for linen, l in enumerate(f):
              idx, utt = l.strip().split(None, 1)
              af = torch.FloatTensor(self.reader[idx]).unsqueeze(1)
              phn = torch.zeros(self.reader[idx].shape[0]).long().unsqueeze(1)
              utt_ids = torch.LongTensor([ self.tgt_vocab.stoi[utt_tok] for utt_tok in utt.split() ]).unsqueeze(1)
              flags = torch.ByteTensor([1, 0]).unsqueeze(1)
              sl = self.reader[idx].shape[0]
              tl = utt_ids.size(0)
              yield af, phn, flags, sl, tl, utt_ids


    def create_batches(self):
       if self.train:
            self.batches = self.pool(self.data())
       else:
            self.batches = []
            for b in self.batch(self.data(), self.batch_size, True):
                self.batches.append(sorted(b))

    """
    def batch(data, batch_size, batch_size_fn=None):
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
        if minibatch:
            yield minibatch
    """

    def batch(self, data, batch_size, pad):
        minibatch = []
        max_len = [0, 0]
        for ex in data:
            max_len[0] = ex[0].size(0) if ex[0].size(0) > max_len[0] else max_len[0]
            max_len[1] = ex[-1].size(0) if ex[-1].size(0) > max_len[1] else max_len[1]
            minibatch.append(ex)
            if len(minibatch) >= batch_size:
                merged_minibatch = []
                for mini_ex in minibatch:
                    af, phn, flag, sl, tl, utt = mini_ex
                    if pad:
                        af = torch.cat([af, torch.zeros(max_len[0] - af.size(0), 1, af.size(2))], dim=0)
                        phn = torch.cat([phn, torch.zeros(max_len[0] - phn.size(0), 1).long()], dim=0)
                        utt = torch.cat([utt, torch.zeros(max_len[1] - utt.size(0), 1).long()], dim=0)
                    else:
                      pass
                    merged_minibatch.append((af, phn, flag, sl, tl, utt))
                yield merged_minibatch
                
                # af_batch, phn_batch, flag_batch, utt_batch = zip(*padded_minibatch)
                # af_batch = torch.cat(af_batch, dim=1)
                # phn_batch = torch.cat(phn_batch, dim=1)
                # flag_batch = torch.cat(flag_batch, dim=1)
                # utt_batch = torch.cat(utt_batch, dim=1)
                # sl_batch, tl_batch = zip(*meta_minibatch)
                # sl_batch = torch.LongTensor(list(sl_batch))
                # tl_batch = torch.LongTensor(list(tl_batch))
                # assert sl_batch.size(0) == tl_batch.size(0) == af_batch.size(1) == phn_batch.size(1) == \
                #        flag_batch.size(1) == utt_batch.size(1)
                # yield af_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch
    
    
    def pool(self, data, random_shuffler=None):
        """Sort within buckets, then batch, then shuffle batches.
        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        if random_shuffler is None:
            random_shuffler = random.shuffle
        print("outer")
        for idx_o, p in enumerate(self.batch(data, self.batch_size * 100, False)):
            print("middle", idx_o)
            p_iter = self.batch(sorted(p, key=lambda p: -p[3]), self.batch_size, True)
            print(p_iter)
            # TODO: how to do random_shuffle
            # for b in random_shuffler(list(p_iter)):
            for b in p_iter:
                print("inner")
                af_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch = zip(*b)
                af_batch = Variable(torch.cat(af_batch, dim=1))
                phn_batch = Variable(torch.cat(phn_batch, dim=1)).unsqueeze(2)
                flag_batch = Variable(torch.cat(flag_batch, dim=1))
                utt_batch = Variable(torch.cat(utt_batch, dim=1)).unsqueeze(2)
                sl_batch = torch.LongTensor(list(sl_batch))
                tl_batch = torch.LongTensor(list(tl_batch))
                assert sl_batch.size(0) == tl_batch.size(0) == af_batch.size(1) == phn_batch.size(1) == \
                       flag_batch.size(1) == utt_batch.size(1)
                yield af_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch
                # yield b

