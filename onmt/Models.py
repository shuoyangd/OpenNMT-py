import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = getattr(nn, opt.rnn_type)(
             input_size, self.hidden_size,
             num_layers=opt.layers,
             dropout=opt.dropout,
             bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs

# Model per discussion w/ Kevin:
# Has two embeddings on the source side -- one to be updated as usual by parallel data,
#       and the other to be a loaded pre-trained embedding and kept constant through
#       out training.
# NOTE:
#   + when input is a tuple, the first element is the actual words, second is the length
#   + last dimension of emb is the hidden dimension
class DualEmbeddingEncoder(nn.Module):

    def __init__(self, opt, dict_, mono_dict):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % (self.num_directions * 2) == 0
        self.hidden_size = opt.rnn_size // self.num_directions // 2 # divide by 2 because of mono and bi
        self.input_size = opt.word_vec_size

        super(DualEmbeddingEncoder, self).__init__()
        self.word_bi = nn.Embedding(dict_.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.word_mono = nn.Embedding(mono_dict.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn_bi = nn.LSTM(self.input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        self.rnn_mono = self.rnn_bi # rnn weights tying
        """
        self.rnn_mono = nn.LSTM(self.input_size, self.hidden_size,
                           num_layers=opt.layers,
                           # weights=self.rnn_bi.weight, # rnn weight tying
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        """

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            if pretrained.size()[1] != self.input_size:
                print("warning: for now the two embeddings should have the same size")
            assert pretrained.size()[1] == self.input_size
            self.word_mono.weight.data.copy_(pretrained)
            self.word_mono.weight.requires_grad = False # fix this embedding

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            bi_emb = pack(self.word_bi(input[0]), lengths)
            mono_emb = pack(self.word_mono(input[0]), lengths)
        else:
            bi_emb = self.word_bi(input)
            mono_emb = self.word_mono(input)
        # output is a namedtuple that has two fields: data, batch_sizes
        # hidden is a tuple (hidden, cell), each is a torch variable
        outputs_bi, hidden_t_bi = self.rnn_bi(bi_emb, hidden)
        outputs_mono, hidden_t_mono = self.rnn_mono(mono_emb, hidden)
        hidden_t = (torch.cat([hidden_t_bi[0], hidden_t_mono[0]], -1),
                    torch.cat([hidden_t_bi[1], hidden_t_mono[1]], -1))
        # unpacked sequence is just a torch variable (holding a floatTensor)
        if isinstance(input, tuple):
            outputs_bi = unpack(outputs_bi)[0]
            outputs_mono = unpack(outputs_mono)[0]
        outputs = torch.cat([outputs_bi, outputs_mono], -1)
        return hidden_t, outputs

class StackedLSTM(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        stackedCell = StackedLSTM if opt.rnn_type == "LSTM" else StackedGRU
        self.rnn = stackedCell(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        if isinstance(enc_hidden, tuple):
            enc_hidden = tuple(self._fix_enc_hidden(enc_hidden[i])
                               for i in range(len(enc_hidden)))
        else:
            enc_hidden = self._fix_enc_hidden(enc_hidden)

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              context, init_output)

        return out
