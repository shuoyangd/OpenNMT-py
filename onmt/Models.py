from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils import weight_norm
import onmt
import pdb
from onmt.Utils import aeq


class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """ A trivial encoder without RNN, just takes mean as final state. """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns. """
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs

class HybridDualEncoderProjection(RNNEncoder):
    def __init__(self, rnn_type, bidirectional, num_audio_layers, num_aug_layers,
                 hidden_size, dropout, embeddings, audio_vec_size,
                 num_concat_flags, add_noise, highway_concat, 
                 do_subsample, do_weight_norm, use_gpu):
      super(HybridDualEncoderProjection, self).__init__(rnn_type, bidirectional, 1,
                                          hidden_size, dropout, embeddings)
      self.num_concat_flags = num_concat_flags 
      self.highway_concat = highway_concat
      self.highway_linear = nn.Linear(hidden_size, hidden_size - num_concat_flags)
      self.add_noise = add_noise
      self.num_audio_layers = num_audio_layers
      self.num_aug_layers = num_aug_layers
      self.do_subsample = do_subsample
      self.do_weight_norm = do_weight_norm
      self.use_proj = True #probably want to make this a arg to pass at cmd line
      self.use_gpu = use_gpu
      assert self.num_aug_layers <= self.num_audio_layers
      assert self.num_audio_layers % self.num_aug_layers == 0
      self.rep = self.num_audio_layers // self.num_aug_layers

      num_directions = 2 if bidirectional else 1
      assert hidden_size % num_directions == 0
      hidden_size = hidden_size // num_directions

      audio_rnn_list = []
      audio_projection_list = []
      self.rnn = None
      for i in range(num_audio_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * (1 if self.use_proj else num_directions)) if i > 0 else audio_vec_size, #embeddings.embedding_size, 
                    hidden_size= (hidden_size - 1) if i == (num_audio_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          audio_rnn_list.append(rnn)
          #projection
          if i < (num_audio_layers - 1) and self.use_proj:
              projection = nn.Linear(hidden_size * num_directions, hidden_size)
          else: 
              projection = None  # we are not applying projection on the last rnn
          audio_projection_list.append(projection)

      self.audio_rnn_list = nn.ModuleList(audio_rnn_list)
      self.audio_projection_list = nn.ModuleList(audio_projection_list)

      aug_rnn_list = []
      aug_projection_list = []
      for i in range(num_aug_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * num_directions) if i > 0 else self.embeddings.embedding_size,
                    hidden_size= (hidden_size - 1) if i == (num_aug_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          aug_rnn_list.append(rnn)
          projection = None # we dont need projection for aug, this is dummy but makes forword code cleaner
          aug_projection_list.append(projection)

      self.aug_rnn_list = nn.ModuleList(aug_rnn_list)
      self.aug_projection_list = nn.ModuleList(aug_projection_list)

    def _check_args(self, input, lengths=None, hidden=None):
        input, is_input_audio, input_flag = input
        s_len, batch, emb_dim = input.size()
        assert isinstance(is_input_audio ,bool)
        # TODO: check with lengths
        # print(input_flag.size())
        # print(input_audio.size())
        assert input_flag.size() == (self.num_concat_flags, batch) 
        if lengths is not None:
            assert s_len == lengths[0]
            #print(s_len, 'sent len', batch, 'batch len', is_input_audio, 'is_audio')
        assert input.size() == (s_len, batch, emb_dim)

    def subsample(self, outputs, lengths):
        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)
        outputs = outputs[::2]
        lengths = [(l + 1) // 2 for l in lengths]
        outputs = pack(outputs, lengths)
        return outputs, lengths
    
    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)
        input, is_input_audio, input_flag = input 
        #print('start batch', is_input_audio)
        if is_input_audio:
            emb = input #(seq_len, batch_size, feats)
        else:
            emb = self.embeddings(input) #(seq_len, batch_size) --> (seq_len, batch_size, feats)
            if self.add_noise:
                n = torch.randn(1, emb.size(1), emb.size(2)).type_as(emb.data)
                n = Variable(n, requires_grad=False) 
                emb += n
                #print('added noise')

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs = packed_emb
        hidden_t = []
        rnn_list = self.audio_rnn_list if is_input_audio else self.aug_rnn_list
        proj_list = self.audio_projection_list if is_input_audio else self.aug_projection_list
        for idx, (rnn,proj) in enumerate(zip(rnn_list, proj_list)):
            #print('rnn', idx, 'is_input_audio', is_input_audio)
            #print('using rnn', rnn)
            outputs, (h_t, c_t) = rnn(outputs, hidden)
            hidden_t.append((h_t, c_t))

            if self.use_proj and proj is not None:
                outputs, lengths = unpack(outputs)
                outputs = proj(outputs)
                outputs = pack(outputs, lengths)
            else:
                pass

            if idx < 2 and self.do_subsample == 1 and is_input_audio:
                outputs, lengths = self.subsample(outputs, lengths)
            else:
                pass # do not sample further

        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)

        if self.num_concat_flags > 0 and self.highway_concat > 0:
            #print(outputs.shape, 'is the output shape hw start')
            assert input_flag.size(0) == self.num_concat_flags
            concat_flag = input_flag.transpose(0,1) #(b,f)
            concat_flag = concat_flag.unsqueeze(0) #(1, b , f)
            concat_flag = concat_flag.expand(outputs.size(0), concat_flag.size(1), concat_flag.size(2))
            outputs = torch.cat([outputs, concat_flag.float()], dim=2)  # (seq_len, batch_size, hidden_size)
            #print(outputs.shape, ' is the output shape hw done')
        else:
            pass       

        #print(outputs.shape, 'is the output shape')
        hl, cl = zip(*hidden_t)
        hl = list(hl)
        cl = list(cl)
        if self.highway_concat > 0:
            ones = Variable(torch.ones(2, hl[-1].size(1), 1).type_as(hl[-1].data) * (1 if is_input_audio else 0))
            hl[-1] = torch.cat([hl[-1], ones], dim=2)
            cl[-1] = torch.cat([cl[-1], ones], dim=2)
        #for ht in hl:
        #    print('hidden sizes', ht.shape)
        #for ct in cl:
        #    print('cell sizes', ct.shape)
        hidden_t = torch.cat(hl, dim=0)  
        cell_t = torch.cat(cl, dim=0)
        if not is_input_audio and self.rep > 1:
            hidden_t = torch.cat([hidden_t] * self.rep, dim = 0)
            cell_t = torch.cat([cell_t] * self.rep, dim = 0)
        # print("done forward", outputs.shape, hidden_t.shape, cell_t.shape)
        return (hidden_t, cell_t), outputs


class HybridDualEncoderProjectionSizeForce(RNNEncoder):
    def __init__(self, rnn_type, bidirectional, num_audio_layers, num_aug_layers,
                 hidden_size, dropout, embeddings, audio_vec_size,
                 num_concat_flags, add_noise, highway_concat, 
                 do_subsample, do_weight_norm, use_gpu):
      super(HybridDualEncoderProjectionSizeForce, self).__init__(rnn_type, bidirectional, 1,
                                          hidden_size, dropout, embeddings)
      self.num_concat_flags = num_concat_flags 
      self.highway_concat = highway_concat
      self.highway_linear = nn.Linear(hidden_size, hidden_size - num_concat_flags)
      self.add_noise = add_noise
      self.num_audio_layers = num_audio_layers
      self.num_aug_layers = num_aug_layers
      self.do_subsample = do_subsample
      self.do_weight_norm = do_weight_norm
      self.use_proj = True #probably want to make this a arg to pass at cmd line
      self.use_gpu = use_gpu
      assert self.num_aug_layers <= self.num_audio_layers
      assert self.num_audio_layers % self.num_aug_layers == 0
      self.rep = self.num_audio_layers // self.num_aug_layers

      num_directions = 2 if bidirectional else 1
      assert hidden_size % num_directions == 0
      hidden_size = hidden_size // num_directions

      self.init_state_projection = nn.Linear(hidden_size, hidden_size // num_directions)

      audio_rnn_list = []
      audio_projection_list = []
      self.rnn = None
      for i in range(num_audio_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * (1 if self.use_proj else num_directions)) if i > 0 else audio_vec_size, #embeddings.embedding_size, 
                    hidden_size= (hidden_size - 1) if i == (num_audio_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          audio_rnn_list.append(rnn)
          #projection
          projection = nn.Linear(hidden_size * num_directions, hidden_size)
          audio_projection_list.append(projection)

      self.audio_rnn_list = nn.ModuleList(audio_rnn_list)
      self.audio_projection_list = nn.ModuleList(audio_projection_list)

      aug_rnn_list = []
      aug_projection_list = []
      for i in range(num_aug_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * num_directions) if i > 0 else self.embeddings.embedding_size,
                    hidden_size= (hidden_size - 1) if i == (num_aug_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          aug_rnn_list.append(rnn)
          projection = nn.Linear(hidden_size * num_directions, hidden_size)
          aug_projection_list.append(projection)

      self.aug_rnn_list = nn.ModuleList(aug_rnn_list)
      self.aug_projection_list = nn.ModuleList(aug_projection_list)

    def _check_args(self, input, lengths=None, hidden=None):
        input, is_input_audio, input_flag = input
        s_len, batch, emb_dim = input.size()
        assert isinstance(is_input_audio ,bool)
        # TODO: check with lengths
        # print(input_flag.size())
        # print(input_audio.size())
        assert input_flag.size() == (self.num_concat_flags, batch) 
        if lengths is not None:
            assert s_len == lengths[0]
            #print(s_len, 'sent len', batch, 'batch len', is_input_audio, 'is_audio')
        assert input.size() == (s_len, batch, emb_dim)

    def subsample(self, outputs, lengths):
        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)
        outputs = outputs[::2]
        lengths = [(l + 1) // 2 for l in lengths]
        outputs = pack(outputs, lengths)
        return outputs, lengths
    
    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)
        input, is_input_audio, input_flag = input 
        #print('start batch', is_input_audio)
        if is_input_audio:
            emb = input #(seq_len, batch_size, feats)
        else:
            emb = self.embeddings(input) #(seq_len, batch_size) --> (seq_len, batch_size, feats)
            if self.add_noise:
                n = torch.randn(1, emb.size(1), emb.size(2)).type_as(emb.data)
                n = Variable(n, requires_grad=False) 
                emb += n
                #print('added noise')

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs = packed_emb
        hidden_t = []
        rnn_list = self.audio_rnn_list if is_input_audio else self.aug_rnn_list
        proj_list = self.audio_projection_list if is_input_audio else self.aug_projection_list
        for idx, (rnn,proj) in enumerate(zip(rnn_list, proj_list)):
            #print('rnn', idx, 'is_input_audio', is_input_audio)
            #print('using rnn', rnn)
            outputs, (h_t, c_t) = rnn(outputs, hidden)
            hidden_t.append((h_t, c_t))

            if self.use_proj and proj is not None:
                outputs, lengths = unpack(outputs)
                outputs = proj(outputs)
                outputs = pack(outputs, lengths)
            else:
                pass

            if idx < 2 and self.do_subsample == 1 and is_input_audio:
                outputs, lengths = self.subsample(outputs, lengths)
            else:
                pass # do not sample further

        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)

        if self.num_concat_flags > 0 and self.highway_concat > 0:
            #print(outputs.shape, 'is the output shape hw start')
            assert input_flag.size(0) == self.num_concat_flags
            concat_flag = input_flag.transpose(0,1) #(b,f)
            concat_flag = concat_flag.unsqueeze(0) #(1, b , f)
            concat_flag = concat_flag.expand(outputs.size(0), concat_flag.size(1), concat_flag.size(2))
            outputs = torch.cat([outputs, concat_flag.float()], dim=2)  # (seq_len, batch_size, hidden_size)
            #print(outputs.shape, ' is the output shape hw done')
        else:
            pass       

        #print(outputs.shape, 'is the output shape')
        hl, cl = zip(*hidden_t)
        hl = list(hl)
        cl = list(cl)
        if self.highway_concat > 0:
            ones = Variable(torch.ones(2, hl[-1].size(1), 1).type_as(hl[-1].data) * (1 if is_input_audio else 0))
            hl[-1] = torch.cat([hl[-1], ones], dim=2)
            cl[-1] = torch.cat([cl[-1], ones], dim=2)
        #for ht in hl:
        #    print('hidden sizes', ht.shape)
        #for ct in cl:
        #    print('cell sizes', ct.shape)
        hidden_t = torch.cat(hl, dim=0)  
        cell_t = torch.cat(cl, dim=0)
        if not is_input_audio and self.rep > 1:
            hidden_t = torch.cat([hidden_t] * self.rep, dim = 0)
            cell_t = torch.cat([cell_t] * self.rep, dim = 0)
        #instead of using a fake initial state, we project the hidden_t to smaller size
        hidden_t = self.init_state_projection(hidden_t) 
        cell_t = self.init_state_projection(cell_t)
        #print("size force encoder: done forward", outputs.shape, hidden_t.shape, cell_t.shape)
        return (hidden_t, cell_t), outputs



class HybridDualEncoder(RNNEncoder):
    def __init__(self, rnn_type, bidirectional, num_audio_layers, num_aug_layers,
                 hidden_size, dropout, embeddings, audio_vec_size,
                 num_concat_flags, add_noise, highway_concat, 
                 do_subsample, do_weight_norm, use_gpu):
      super(HybridDualEncoder, self).__init__(rnn_type, bidirectional, 1,
                                          hidden_size, dropout, embeddings)
      self.num_concat_flags = num_concat_flags 
      self.highway_concat = highway_concat
      self.highway_linear = nn.Linear(hidden_size, hidden_size - num_concat_flags)
      self.add_noise = add_noise
      self.num_audio_layers = num_audio_layers
      self.num_aug_layers = num_aug_layers
      self.do_subsample = do_subsample
      self.do_weight_norm = do_weight_norm
      self.use_gpu = use_gpu
      assert self.num_aug_layers <= self.num_audio_layers
      assert self.num_audio_layers % self.num_aug_layers == 0
      self.rep = self.num_audio_layers // self.num_aug_layers

      num_directions = 2 if bidirectional else 1
      assert hidden_size % num_directions == 0
      hidden_size = hidden_size // num_directions

      audio_rnn_list = []
      self.rnn = None
      for i in range(num_audio_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * num_directions) if i > 0 else audio_vec_size, #embeddings.embedding_size, 
                    hidden_size= (hidden_size - 1) if i == (num_audio_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          #if self.do_weight_norm == 1:
          #    if len(self.use_gpu) > 0:
          #        rnn = rnn.cuda()
          #    rnn = weight_norm(rnn, name='weight_ih_l0')
          #    rnn = weight_norm(rnn, name='weight_hh_l0')
          audio_rnn_list.append(rnn)
      self.audio_rnn_list = nn.ModuleList(audio_rnn_list)

      aug_rnn_list = []
      for i in range(num_aug_layers):
          rnn = getattr(nn, rnn_type)(
                    input_size= (hidden_size * num_directions) if i > 0 else self.embeddings.embedding_size,
                    hidden_size= (hidden_size - 1) if i == (num_aug_layers - 1) and self.highway_concat == 1 else hidden_size, #if i < (num_layers - 1) else (hidden_size + (num_concat_flags * use_highway_concat)),
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional)
          aug_rnn_list.append(rnn)
      self.aug_rnn_list = nn.ModuleList(aug_rnn_list)

    def _check_args(self, input, lengths=None, hidden=None):
        input, is_input_audio, input_flag = input
        s_len, batch, emb_dim = input.size()
        assert isinstance(is_input_audio ,bool)
        # TODO: check with lengths
        # print(input_flag.size())
        # print(input_audio.size())
        assert input_flag.size() == (self.num_concat_flags, batch) 
        if lengths is not None:
            assert s_len == lengths[0]
            #print(s_len, 'sent len', batch, 'batch len', is_input_audio, 'is_audio')
        assert input.size() == (s_len, batch, emb_dim)

    def subsample(self, outputs, lengths):
        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)
        outputs = outputs[::2]
        lengths = [(l + 1) // 2 for l in lengths]
        outputs = pack(outputs, lengths)
        return outputs, lengths

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)
        input, is_input_audio, input_flag = input 
        #print('start batch', is_input_audio)
        if is_input_audio:
            emb = input #(seq_len, batch_size, feats)
        else:
            emb = self.embeddings(input) #(seq_len, batch_size) --> (seq_len, batch_size, feats)
            if self.add_noise:
                n = torch.randn(1, emb.size(1), emb.size(2)).type_as(emb.data)
                n = Variable(n, requires_grad=False) 
                emb += n
                #print('added noise')

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs = packed_emb
        hidden_t = []
        rnn_list = self.audio_rnn_list if is_input_audio else self.aug_rnn_list
        for idx, rnn in enumerate(rnn_list):
            #print('rnn', idx, 'is_input_audio', is_input_audio)
            #print('using rnn', rnn)
            outputs, (h_t, c_t) = rnn(outputs, hidden)
            hidden_t.append((h_t, c_t))
            if idx < 2 and self.do_subsample == 1 and is_input_audio:
                outputs, lengths = self.subsample(outputs, lengths)
            else:
                pass # do not sample further

        if lengths is not None and not self.no_pack_padded_seq:
            outputs, lengths = unpack(outputs)

        if self.num_concat_flags > 0 and self.highway_concat > 0:
            #print(outputs.shape, 'is the output shape hw start')
            assert input_flag.size(0) == self.num_concat_flags
            concat_flag = input_flag.transpose(0,1) #(b,f)
            concat_flag = concat_flag.unsqueeze(0) #(1, b , f)
            concat_flag = concat_flag.expand(outputs.size(0), concat_flag.size(1), concat_flag.size(2))
            outputs = torch.cat([outputs, concat_flag.float()], dim=2)  # (seq_len, batch_size, hidden_size)
            #print(outputs.shape, ' is the output shape hw done')
        else:
            pass       

        #print(outputs.shape, 'is the output shape')
        hl, cl = zip(*hidden_t)
        hl = list(hl)
        cl = list(cl)
        if self.highway_concat > 0:
            ones = Variable(torch.ones(2, hl[-1].size(1), 1).type_as(hl[-1].data) * (1 if is_input_audio else 0))
            hl[-1] = torch.cat([hl[-1], ones], dim=2)
            cl[-1] = torch.cat([cl[-1], ones], dim=2)
        #for ht in hl:
        #    print('hidden sizes', ht.shape)
        #for ct in cl:
        #    print('cell sizes', ct.shape)
        hidden_t = torch.cat(hl, dim=0)  
        cell_t = torch.cat(cl, dim=0)
        if not is_input_audio and self.rep > 1:
            hidden_t = torch.cat([hidden_t] * self.rep, dim = 0)
            cell_t = torch.cat([cell_t] * self.rep, dim = 0)
        #print('done', is_input_audio, outputs.shape, 'outputs size')
        return (hidden_t, cell_t), outputs


class RNNDecoderBase(nn.Module):
    """
    RNN decoder base class.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.ContextGateFactory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = \
            self._run_forward_pass(input, context, state)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Attention.
    Currently no 'coverage_attn' and 'copy_attn' support.
    """
    def _run_forward_pass(self, input, context, state):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Input Feed and Attention.
    """
    def _run_forward_pass(self, input, context, state):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(rnn_output,
                                          context.transpose(0, 1))
            if self.context_gate is not None:
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

class InputFeedRNNDecoderWithFlags(InputFeedRNNDecoder):
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings, num_concat_flags, use_highway_concat):
        super(InputFeedRNNDecoderWithFlags, self).__init__(rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings)
       
        #overwirte self.attn with new GlobalAttentionWithFlags
        print("GlobalAttentionWithFlags overwrite")
        self.attn = onmt.modules.GlobalAttentionWithFlags(
            dim = hidden_size, 
            dim_with_flags = hidden_size + (num_concat_flags * use_highway_concat),
            coverage=coverage_attn,
            attn_type=attn_type
        )


class NMTModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_hidden = tuple([enc_hidden[0][-2 * self.decoder.num_layers:], enc_hidden[1][-2 * self.decoder.num_layers:]])
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
