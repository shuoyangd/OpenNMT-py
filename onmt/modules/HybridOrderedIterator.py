#import sys
#import codecs
#import argparse
import lazy_io
import math
import random
import torch
from torch.autograd import Variable
import numpy as np

class HybridOrderedIterator:
    def __init__(self, train_mode, batch_size, audio_file, augmenting_file, vocab_file,  augmenting_data_names, mix_factor, embedding_size, device):
      self.train_mode = train_mode
      self.batch_size = batch_size
      self.audio_src_reader = lazy_io.read_dict_scp(audio_file + '.src' )
      self.audio_tgt_reader = open(audio_file + '.tgt', 'r')
      self.tgt_vocab = torch.load(vocab_file)[1][1]
      self.augmenting_src_vocab = torch.load(vocab_file)[0][1] 
      self.augmenting_src_reader = open(augmenting_file + '.src', 'r')
      self.augmenting_tgt_reader = open(augmenting_file + '.tgt', 'r')
      assert isinstance(augmenting_data_names, list)
      self.augmenting_data_names = {name: (idx+1) for idx, name in enumerate(augmenting_data_names)}
      self.device = device
      self.embedding_size = embedding_size
      self.mix_factor = mix_factor


    def __len__(self):
      return math.ceil(3696 / self.batch_size)

    def __iter__(self):
      self.create_batches()
      for batch in self.batches:
        yield batch
    
    def get_audio_pair(self, audio_tgt_line):
        if audio_tgt_line is not None:
            idx, audio_tgt = audio_tgt_line.strip().split(None, 1)
            audio_src = self.audio_src_reader.get(idx, None)
            return audio_src, audio_tgt 
        else:
            return None, None

    def _check_end(self, audio_pair, augmenting_pair):
        #is_done = [i[0] is not  None for i in pairs] #True if pair is not None
        #return functools.reduce(lambda x,y: x or y, is_done) # True if any item in is_done is True
        return (audio_pair != (None, None)) or (augmenting_pair != (None, None))

    def _make_tensors(self, pair, is_audio):
        src, tgt = pair
        flags = torch.zeros(len(self.augmenting_data_names) + 1).byte()
        if is_audio: 
            assert isinstance(src, np.ndarray) #(seq_len, features)
            assert isinstance(tgt, str)
            audio_src = torch.FloatTensor(src).unsqueeze(1) #(seq_len, 1, features)
            aug_src = torch.zeros((1, 1, self.embedding_size)).long() #fake (1, 1, features)
            tgt = ["<s>"] + tgt.split() + ["</s>"]
            tgt = torch.LongTensor([self.tgt_vocab.stoi[tok] for tok in tgt]).unsqueeze(1) #(seq_len, 1)
            flags[0] = 1
            flags = flags.unsqueeze(1)
            src_length = audio_src.size(0)
        else:
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            audio_src = torch.zeros((1, 1, self.embedding_size)).float()  #fake (1, 1, features)
            src = ["<s>"] + src.split() + ["</s>"]
            aug_src = torch.LongTensor([self.augmenting_src_vocab.stoi[tok] for tok in src]).unsqueeze(1)
            aug_name, tgt = tgt.strip().split(None, 1)
            tgt = ["<s>"] + tgt.split() + ["</s>"]
            tgt = torch.LongTensor([self.tgt_vocab.stoi[tok] for tok in tgt]).unsqueeze(1)
            flags[self.augmenting_data_names[aug_name]] = 1
            flags = flags.unsqueeze(1)
            src_length = aug_src.size(0)
        tgt_length = tgt.size(0)
        return audio_src, aug_src, flags, src_length, tgt_length, tgt


    def data(self):
        audio_pair = self.get_audio_pair(self.audio_tgt_reader.readline())
        aug_pair = (self.augmenting_src_reader.readline(), self.augmenting_tgt_reader.readline())
        while self._check_end(audio_pair, aug_pair):
            if (not self.train_mode) or (np.random.rand() > self.mix_factor and audio_pair != (None, None)): # do a audio example
                yield self._make_tensors(audio_pair, True)
            elif aug_pair != (None, None):
                yield self._make_tensors(aug_pair, False)
            else:
                pass

    """
    def data(self):
       with open(self.utts_file, "r") as f:
          for linen, l in enumerate(f):
              idx, utt = l.strip().split(None, 1)
              utt_toks = ["<s>"] + utt.split() + ["</s>"]
              utt_ids = torch.LongTensor([ self.tgt_vocab.stoi[utt_tok] for utt_tok in utt_toks ]).unsqueeze(1)
              val = self.audio_src_reader.get(idx, None)
              if val is not None:
                  af = torch.FloatTensor(val).unsqueeze(1)
                  flags = torch.ByteTensor([1, 0]).unsqueeze(1)
                  phn = torch.zeros(self.audio_src_reader[idx].shape[0]).long().unsqueeze(1)
                  sequence_length = af.size(0)
              else:
                  utt_src_line = self.mono_src.readline()
                  assert utt_src_line is not None
                  flags = torch.ByteTensor([0, 1]).unsqueeze(1)
                  phn = torch.LongTensor([ self.tgt_vocab.stoi[utt_tok] for utt_tok in utt_toks ]).unsqueeze(1)
                  af = torch.rand(1, 1, self.embedding_size)
                  sequence_length = phn.size(0)
              sl = sequence_length
              tl = utt_ids.size(0)
              yield af, phn, flags, sl, tl, utt_ids
    """

    def create_batches(self):
       if self.train_mode:
            self.batches = self.pool(self.data())
       else:
            self.batches = self.pool(self.data(), False, 1)

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
            audio_src_, phn_, flag_, sl_, tl_, utt_ = ex
            max_len[0] = audio_src_.size(0) if audio_src_.size(0) > max_len[0] else max_len[0]
            max_len[0] = phn_.size(0) if phn_.size(0) > max_len[0] else max_len[0]
            max_len[1] = utt_.size(0) if utt_.size(0) > max_len[1] else max_len[1]
            minibatch.append(ex)
            if len(minibatch) >= batch_size:
                merged_minibatch = []
                for mini_ex in minibatch:
                    audio_src, phn, flag, sl, tl, utt = mini_ex
                    if pad:
                        audio_src = torch.cat([audio_src, torch.zeros(max_len[0] - audio_src.size(0), 1, audio_src.size(2))], dim=0)
                        phn = torch.cat([phn, torch.zeros(max_len[0] - phn.size(0), 1).long()], dim=0)
                        utt = torch.cat([utt, torch.zeros(max_len[1] - utt.size(0), 1).long()], dim=0)
                    else:
                      pass
                    merged_minibatch.append((audio_src, phn, flag, sl, tl, utt))
                minibatch = []
                yield merged_minibatch

        if len(minibatch) > 0:
            merged_minibatch = []
            for mini_ex in minibatch:
                audio_src, phn, flag, sl, tl, utt = mini_ex
                if pad:
                    #TODO: use pytorch's built in padder torch.nn.utils.rnn.pad_sequence
                    audio_src = torch.cat([audio_src, torch.zeros(max_len[0] - audio_src.size(0), 1, audio_src.size(2))], dim=0)
                    phn = torch.cat([phn, torch.zeros(max_len[0] - phn.size(0), 1).long()], dim=0)
                    utt = torch.cat([utt, torch.zeros(max_len[1] - utt.size(0), 1).long()], dim=0)
                else:
                  pass
                merged_minibatch.append((audio_src, phn, flag, sl, tl, utt))
            minibatch = []
            yield merged_minibatch
                
                # audio_src_batch, phn_batch, flag_batch, utt_batch = zip(*padded_minibatch)
                # audio_src_batch = torch.cat(audio_src_batch, dim=1)
                # phn_batch = torch.cat(phn_batch, dim=1)
                # flag_batch = torch.cat(flag_batch, dim=1)
                # utt_batch = torch.cat(utt_batch, dim=1)
                # sl_batch, tl_batch = zip(*meta_minibatch)
                # sl_batch = torch.LongTensor(list(sl_batch))
                # tl_batch = torch.LongTensor(list(tl_batch))
                # assert sl_batch.size(0) == tl_batch.size(0) == audio_src_batch.size(1) == phn_batch.size(1) == \
                #        flag_batch.size(1) == utt_batch.size(1)
                # yield audio_src_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch
    
    
    def pool(self, data, do_shuffle=True, bucket_factor = 100):
        """Sort within buckets, then batch, then shuffle batches.
        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        random_shuffler = random.shuffle
        # print("outer")
        for idx_o, p in enumerate(self.batch(data, self.batch_size * bucket_factor, False)):
            # print("middle", idx_o)
            # print("type(p) = " + str(type(p)))
            # print("len(p) = " + str(len(p)))
            p_iter = self.batch(sorted(p, key=lambda x: -x[3]), self.batch_size, True)
            # p_iter = self.batch(p, self.batch_size, True)
            # TODO: how to do random_shuffle
            # for b in random_shuffler(list(p_iter)):
            p_list = list(p_iter)
            # print(len(p_list))
            if do_shuffle:
              random_shuffler(p_list)
            # assert len(p_list) == 100
            # assert len(p_list[0]) == 8
            # assert len(p_list[0][0]) == 6
            # for b in random_shuffler(list(p_iter)):
            for b in p_list:
                # print("inner")
                audio_src_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch = zip(*b)
                audio_src_batch = Variable(torch.cat(audio_src_batch, dim=1))
                phn_batch = Variable(torch.cat(phn_batch, dim=1)).unsqueeze(2)
                flag_batch = Variable(torch.cat(flag_batch, dim=1))
                utt_batch = Variable(torch.cat(utt_batch, dim=1)).unsqueeze(2)
                sl_batch = torch.LongTensor(list(sl_batch))
                tl_batch = torch.LongTensor(list(tl_batch))
                # assert sl_batch.size(0) == tl_batch.size(0) == audio_src_batch.size(1) == phn_batch.size(1) == \
                #        flag_batch.size(1) == utt_batch.size(1)
                if self.device != -1: 
                    audio_src_batch = audio_src_batch.cuda()
                    phn_batch = phn_batch.cuda() 
                    flag_batch = flag_batch.cuda()
                    sl_batch = sl_batch.cuda()
                    tl_batch = tl_batch.cuda()
                    utt_batch = utt_batch.cuda()
                yield audio_src_batch, phn_batch, flag_batch, sl_batch, tl_batch, utt_batch
                # yield b

