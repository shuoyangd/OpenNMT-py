import lazy_io
import math
import random
import torch
from torch.autograd import Variable
from collections import namedtuple

ExInstance = namedtuple('ExInstance', 'src is_audio flags src_length tgt_length tgt')
class HybridOrderedIterator:
    def __init__(self, train_mode, batch_size, audio_file, augmenting_file, tgt_vocab, src_vocab,  augmenting_data_names, init_mix_factor, end_mix_factor,
            num_aug_instances, num_audio_instances, num_epochs, device):
      self.train_mode = train_mode
      self.batch_size = batch_size
      self.num_epochs = num_epochs
      self.num_audio_instances =  num_audio_instances
      self.audio_src_reader_file = audio_file + '.src'
      self.audio_src_reader = None
      self.audio_tgt_reader_file = audio_file + '.tgt'
      self.audio_tgt_reader = None
      self.tgt_vocab = tgt_vocab #torch.load(vocab_file)[1][1]
      #self.aug_data_names = {name: (idx+1) for idx, name in enumerate(augmenting_data_names)}
      self.flags_size = 2 #len(augmenting_data_names) + 1
      self.mix_factor = init_mix_factor
      self.end_mix_factor = end_mix_factor
      self.mix_step = (self.mix_factor - self.end_mix_factor) / float(num_epochs)
      self.device = device
      self.use_aug = self.train_mode and (self.mix_factor > 0.0) and (augmenting_file is not None)
      if self.use_aug:
          self.aug_src_vocab = src_vocab #torch.load(vocab_file)[0][1] 
          self.aug_src_reader_file = augmenting_file + '.src'
          self.aug_tgt_reader_file = augmenting_file + '.tgt'
          self.num_aug_instances = num_aug_instances
      else:
          self.aug_src_vocab = None #torch.load(vocab_file)[0][1] 
          self.aug_src_reader_file = None
          self.aug_tgt_reader_file = None
          self.mix_factor = 0.0
          self.num_aug_instances = 0 #num_aug_instances
      self.epoch_counter = 0

    def init_audio_reader(self,):
       print('initializing audio reader.. train_mode:', self.train_mode) 
       self.audio_src_reader = lazy_io.read_dict_scp(self.audio_src_reader_file)
       self.audio_tgt_reader = open(self.audio_tgt_reader_file, 'r')

    def init_aug_reader(self,):
       print('initializing aug reader... train_mode:', self.train_mode) 
       self.aug_src_reader = open(self.aug_src_reader_file, 'r')
       self.aug_tgt_reader = open(self.aug_tgt_reader_file, 'r')



    def create_batches(self):
       #initialize file readers to starting point
       self.init_audio_reader()
       if self.train_mode:
           self.epoch_counter += 1
           self.mix_factor -= self.mix_step
           self.mix_factor = self.mix_factor if self.mix_factor > self.end_mix_factor else self.end_mix_factor
           print('creating batch epoch:%d mix_factor:%.4f' %(self.epoch_counter, self.mix_factor))
           if self.use_aug:
               self.init_aug_reader()
               mf = self.mix_factor
           else:
               mf = 0.0
           self.batches = self.pool(self.audio_data(), self.augment_data(), mix_factor = mf, do_shuffle = True, bucket_factor = 100)
       else:
           self.batches = self.pool(self.audio_data(), self.none_iter(), mix_factor = 0.0, do_shuffle = False, bucket_factor = 1) #buckect_factor =1 ensures order of batches is unchanged.

    def __len__(self):
      s = (1.0 / (1.0 - self.mix_factor)) * self.num_audio_instances
      return math.ceil(s / self.batch_size)

    def __iter__(self):
      self.create_batches()
      for batch in self.batches:
        yield batch

    def get_next_aug_pair(self):
        if self.use_aug:
            aug_src_line = self.aug_src_reader.readline()
            aug_tgt_line = self.aug_tgt_reader.readline()
            src_ok = aug_src_line is not None and aug_src_line.strip() != ''
            tgt_ok = aug_tgt_line is not None and aug_tgt_line.strip() != ''
            if src_ok and tgt_ok:
                return aug_src_line, aug_tgt_line
            else:
                print('end of aug reader...', self.train_mode)
                self.init_aug_reader() #re initialize and loop back
                aug_src_line = self.aug_src_reader.readline()
                aug_tgt_line = self.aug_tgt_reader.readline()
                return aug_src_line, aug_tgt_line
        else:
            return None, None

    def get_next_audio_pair(self):
        audio_tgt_line = self.audio_tgt_reader.readline()
        if audio_tgt_line is not None and audio_tgt_line.strip() != '':
            idx, audio_tgt = audio_tgt_line.strip().split(None, 1)
            audio_src = self.audio_src_reader[idx]
            return audio_src, audio_tgt 
        else:
            print('end of audio reader...', self.train_mode)
            return None, None

    def is_end(self, audio_pair): #, aug_pair):
        is_audio_end = audio_pair[1] == None
        return is_audio_end

    def _make_tensors(self, pair, is_audio):
        src, tgt = pair
        flags = torch.zeros(self.flags_size).byte()
        if is_audio: 
            src = torch.FloatTensor(src).unsqueeze(1) #(seq_len, 1, features)
            tgt = ["<s>"] + tgt.split() + ["</s>"]
            tgt = torch.LongTensor([self.tgt_vocab.stoi[tok] for tok in tgt]).unsqueeze(1) #(seq_len, 1)
            flags[0] = 1
            flags = flags.unsqueeze(1)
            src_length = src.size(0)
        else:
            src = ["<s>"] + src.split() + ["</s>"]
            src = torch.LongTensor([self.aug_src_vocab.stoi[tok] for tok in src]).unsqueeze(1).unsqueeze(2) #(seg_len, 1, 1)
            aug_name, tgt = tgt.strip().split(None, 1)
            tgt = ["<s>"] + tgt.split() + ["</s>"]
            tgt = torch.LongTensor([self.tgt_vocab.stoi[tok] for tok in tgt]).unsqueeze(1)
            flags[1] = 1
            flags = flags.unsqueeze(1)
            src_length = src.size(0)
        tgt_length = tgt.size(0)
        ex_instance = ExInstance(src, is_audio, flags, src_length, tgt_length, tgt)
        return ex_instance

    def augment_data(self,):
        aug_pair = self.get_next_aug_pair()
        while aug_pair[1] is not None:
            if not self.train_mode:
                raise BaseException("aug data should not be called in validation and test mode")
            tmp = self._make_tensors(aug_pair, False)
            aug_pair = self.get_next_aug_pair()
            yield tmp

    def audio_data(self):
        audio_pair = self.get_next_audio_pair()
        while not self.is_end(audio_pair): #, aug_pair):
            tmp = self._make_tensors(audio_pair, True)
            audio_pair = self.get_next_audio_pair() 
            yield tmp

    def none_iter(self):
        while True:
            yield None

    def batch(self, data, batch_size, pad, adapt_batch_size):
        max_length_in = 450  #tuned by hand for batch size of 64
        max_length_out = 80  #tuned by hand for batch size of 64 
        minibatch = []
        max_len = [0, 0]
        for ex in data:
            src_, is_audio_, flag_, sl_, tl_, tgt_ = ex
            max_len[0] = src_.size(0) if src_.size(0) > max_len[0] else max_len[0]
            max_len[1] = tgt_.size(0) if tgt_.size(0) > max_len[1] else max_len[1]
            if adapt_batch_size:
                factor = max(int(max_len[0] / max_length_in), int(max_len[1] / max_length_out))
                #factor = factor ** 1.2
                current_batch_limit = max(1, int(batch_size / (1 + factor)))
            else:
                current_batch_limit = batch_size
            minibatch.append(ex)
            if len(minibatch) >= current_batch_limit:
                merged_minibatch = []
                for mini_ex in minibatch:
                    src, is_audio, flag, sl, tl, tgt = mini_ex
                    if pad:
                        pad_idx = 0 if is_audio else self.aug_src_vocab.stoi['<pad>']
                        src_padder = torch.ones(max_len[0] - src.size(0), 1, src.size(2)).type_as(src) * pad_idx
                        src = torch.cat([src, src_padder], dim=0)
                        tgt_padder = torch.ones(max_len[1] - tgt.size(0), 1).type_as(tgt) * self.tgt_vocab.stoi['<pad>']
                        tgt = torch.cat([tgt, tgt_padder], dim=0)
                    else:
                      pass
                    ex_instance_padded = ExInstance(src, is_audio, flag, sl, tl, tgt)
                    merged_minibatch.append(ex_instance_padded) 
                minibatch = []
                #print('max_len', max_len, current_batch_limit)
                if pad and adapt_batch_size:
                    #print(self.train_mode, 'current_batch_limit', current_batch_limit, 'max_lens', max_len[0], max_len[1], 'factor', factor)
                    pass
                max_len = [0, 0]
                yield merged_minibatch

        if len(minibatch) > 0:
            merged_minibatch = []
            for mini_ex in minibatch:
                src, is_audio, flag, sl, tl, tgt = mini_ex
                if pad:
                    pad_idx = 0 if is_audio else self.aug_src_vocab.stoi['<pad>']
                    src_padder = torch.ones(max_len[0] - src.size(0), 1, src.size(2)).type_as(src) * pad_idx
                    src = torch.cat([src, src_padder], dim=0)
                    tgt_padder = torch.ones(max_len[1] - tgt.size(0), 1).type_as(tgt) * self.tgt_vocab.stoi['<pad>']
                    tgt = torch.cat([tgt, tgt_padder], dim=0)
                else:
                  pass
                ex_instance_padded = ExInstance(src, is_audio, flag, sl, tl, tgt)
                merged_minibatch.append(ex_instance_padded) 
            minibatch = []
            if pad and adapt_batch_size:
                #print(self.train_mode, 'final_batch_limit', len(merged_minibatch), 'max_lens', max_len[0], max_len[1],'factor', None)
                pass
            max_len = [0, 0]
            yield merged_minibatch

    def pool(self, audio_iter, aug_iter, mix_factor, do_shuffle, bucket_factor): #TODO: bucket_factor should be small for toy data
        """Sort within buckets, then batch, then shuffle batches.
        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        processed = 0
        random_shuffler = random.shuffle
        audio_bucket_factor = int((1.0 - mix_factor) * (bucket_factor))
        aug_bucket_factor = int(mix_factor *  (bucket_factor))
        audio_bucket_iter = self.batch(audio_iter, self.batch_size * audio_bucket_factor, False, False)
        if aug_iter is not None and aug_bucket_factor > 0:
            augment_bucket_iter = self.batch(aug_iter, self.batch_size * aug_bucket_factor, False, False)
        else:
            print('None iter used')
            augment_bucket_iter = self.none_iter() #some iter that returns None forever
        for  aug_bucket, audio_bucket in zip(augment_bucket_iter, audio_bucket_iter):  #in enumerate(concat_batch):
            if audio_bucket is not None:
                sorted_audio_bucket = sorted(audio_bucket, key=lambda x: -x.src_length)
                audio_p_iter = self.batch(sorted_audio_bucket, self.batch_size, pad = True, adapt_batch_size= True)
                audio_p_list = list(audio_p_iter)
            else:
                audio_p_list = []
                #print('ending pool loop')
                #break # the loop ends once audio data is completed
            if aug_bucket is not None:
                sorted_aug_bucket = sorted(aug_bucket, key=lambda x: -x.src_length)
                aug_p_iter = self.batch(sorted_aug_bucket, self.batch_size, pad = True, adapt_batch_size= True)
                aug_p_list = list(aug_p_iter)
            else:
                aug_p_list = []
            p_list = aug_p_list + audio_p_list
            if do_shuffle:
              random_shuffler(p_list)
            for b in p_list:
                src_batch, is_audio_batch, flag_batch, sl_batch, tl_batch, tgt_batch = zip(*b)
                flag_batch = Variable(torch.cat(flag_batch, dim=1), volatile=(not self.train_mode))
                assert flag_batch.data[0][0] == flag_batch.data[0][-1]
                assert flag_batch.data[1][0] == flag_batch.data[1][-1]
                if flag_batch.data[0][0] == 1:
                    assert is_audio_batch[0]
                    is_audio = True
                    src_batch = Variable(torch.cat(src_batch, dim=1), volatile=(not self.train_mode))
                else:
                    assert not is_audio_batch[0]
                    is_audio = False
                    src_batch = Variable(torch.cat(src_batch, dim=1), volatile=(not self.train_mode))
                tgt_batch = Variable(torch.cat(tgt_batch, dim=1), volatile=(not self.train_mode)).unsqueeze(2)
                sl_batch = torch.LongTensor(list(sl_batch))
                tl_batch = torch.LongTensor(list(tl_batch))
                if self.device != -1: 
                    src_batch = src_batch.cuda() 
                    flag_batch = flag_batch.cuda()
                    sl_batch = sl_batch.cuda()
                    tl_batch = tl_batch.cuda()
                    tgt_batch = tgt_batch.cuda()
                processed += src_batch.size(1)
                #print(self.train_mode, is_audio, 'src_dim', src_batch.shape, 'tgt_dim', tgt_batch.shape, 'processed', processed)
                yield ExInstance(src_batch, is_audio, flag_batch, sl_batch, tl_batch, tgt_batch)
