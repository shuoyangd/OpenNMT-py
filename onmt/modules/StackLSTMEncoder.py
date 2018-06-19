from enum import Enum
import logging
import onmt.modules.utils.tensor
import onmt.modules.utils.rand

import pdb

import torch
from torch import nn
# from torch.autograd import Variable
from onmt.modules.StackLSTMCell import StackLSTMCell
from onmt.modules.StateStack import StateStack
from onmt.modules.MultiLayerLSTMCell import MultiLayerLSTMCell

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)


class TransitionSystems(Enum):
  ASd = 0
  AER = 1
  AES = 2
  AH = 3
  ASw = 4

  @classmethod
  def has_value(cls, value):
    return any(value == item for item in cls)


AER_map = {"Shift": (1, -1), "Reduce": (-1, 0), "Left-Arc": (-1, 0), "Right-Arc": (1, -1)}
AH_map = {"Shift": (1, -1), "Left-Arc": (-1, 0), "Right-Arc": (-1, 0)}


class StackLSTMEncoder(onmt.Models.EncoderBase):

  def __init__(self, num_layers, transSys, stack_size, gpuid,
               action_emb_dim, hid_dim, state_dim=None, dropout_rate=0.0, embeddings=None, use_bridge=False, gumbel_temprature=1., gumbel_beta=1.):
    """
    :param word_emb, with type nn.Embeddings
    :param actions: a python list of actions, denoted as "transition" (unlabled) or "transition|label" (labeled).
           e.g. "Left-Arc", "Left-Arc|nsubj"
    :param pre_vocab: vocabulary with pre-trained word embedding, with type torchtext.vocab.Vocab.
           This vocabulary should contain the word embedding itself as well, see torchtext documents for details.
    :param postags: a python list of pos, in strings (optional)
    """

    super(StackLSTMEncoder, self).__init__()
    self.transSys = TransitionSystems(transSys)
    self.use_bridge = use_bridge
    self.gumbel_temperature = gumbel_temprature
    self.gumbel_beta = gumbel_beta

    # gpu
    self.gpuid = gpuid
    self.dtype = torch.cuda.FloatTensor if len(gpuid) >= 1 else torch.FloatTensor
    self.long_dtype = torch.cuda.LongTensor if len(gpuid) >= 1 else torch.LongTensor

    # vocabularies
    # assuming <pad> is always at the end
    self.actions = ["Shift", "Left-Arc", "Right-Arc", "<pad>"]  # FIXME: temporarily, for debugging

    # action mappings
    self.stack_action_mapping = torch.zeros(len(self.actions),).type(self.long_dtype)
    self.buffer_action_mapping = torch.zeros(len(self.actions),).type(self.long_dtype)
    self.set_action_mappings()

    # embeddings
    self.word_emb = embeddings
    self.action_emb = nn.Embedding(len(self.actions), action_emb_dim)

    state_dim = hid_dim if not state_dim else state_dim
    self.hid_dim = hid_dim
    self.action_emb_dim = action_emb_dim
    self.input_dim = embeddings.embedding_size
    self.num_layers = num_layers
    self.stack_size = stack_size
    self.root_input = nn.Parameter(torch.rand(self.input_dim))

    # recurrent components
    # the only reason to have unified h0 and c0 is because pre_buffer and buffer should have the same initial states
    # but due to simplicity we just borrowed this for all three recurrent components (stack, buffer, action)
    self.h0 = nn.Parameter(torch.rand(hid_dim,).type(self.dtype))
    self.c0 = nn.Parameter(torch.rand(hid_dim,).type(self.dtype))
    # FIXME: there is no dropout in StackLSTMCell at this moment
    # BufferLSTM could have 0 or 2 parameters, depending on what is passed for initial hidden and cell state
    self.stack = StackLSTMCell(self.input_dim, hid_dim, dropout_rate, stack_size, num_layers, self.h0, self.c0)
    self.buffer = StateStack(hid_dim, self.h0)
    self.token_stack = StateStack(self.input_dim)
    self.token_buffer = StateStack(self.input_dim) # elememtns in this buffer has size input_dim so h0 and c0 won't fit
    self.pre_buffer = MultiLayerLSTMCell(input_size=self.input_dim, hidden_size=hid_dim, num_layers=num_layers) # FIXME: dropout needs to be implemented manually
    self.history = MultiLayerLSTMCell(input_size=action_emb_dim, hidden_size=hid_dim, num_layers=num_layers) # FIXME: dropout needs to be implemented manually

    self.summarize_states = nn.Sequential(
        nn.Linear(3 * hid_dim, state_dim),
        nn.ReLU()
    )
    self.state_to_actions = nn.Linear(state_dim, len(self.actions))

    # self.softmax = nn.LogSoftmax(dim=-1) # LogSoftmax works on final dimension only for two dimension tensors. Be careful.
    if self.use_bridge:
      self._initialize_bridge(rnn_type,
                              hidden_size,
                              num_layers)


  def forward(self, inputs, lengths=None, encoder_state=None):
    """

    :param inputs: (seq_len, batch_size) tokens of the input sentence, preprocessed as vocabulary indexes
    :param tokens_mask: (seq_len, batch_size) indication of whether this is a "real" token or a padding
    :param postags: (seq_len, batch_size) optional POS-tag of the tokens of the input sentence, preprocessed as indexes
    :param actions: (seq_len, batch_size) optional golden action sequence, preprocessed as indexes
    :return: output: (output_seq_len, batch_size, len(actions)), or when actions = None, (output_seq_len, batch_size, max_step_length)
             encodings: (output_seq_len, batch_size, state_dim)
    """

    tokens_mask = onmt.modules.utils.tensor.make_mask(lengths).transpose(0, 1).type(self.long_dtype).byte()
    seq_len = inputs.size()[0]
    batch_size = inputs.size()[1]
    emb = self.word_emb(inputs.t()) # (seq_len, batch_size, word_emb_dim)
    emb = emb.transpose(0, 1)
    emb_rev = onmt.modules.utils.tensor.masked_revert(emb, tokens_mask.unsqueeze(2).expand(seq_len, batch_size, self.input_dim), 0) # (seq_len, batch_size, input_dim), pad at end

    # initialize stack
    self.stack.build_stack(batch_size, self.gpuid)

    # initialize and preload buffer
    buffer_hiddens = torch.zeros((seq_len, batch_size, self.hid_dim * self.num_layers)).type(self.dtype)
    buffer_cells = torch.zeros((seq_len, batch_size, self.hid_dim * self.num_layers)).type(self.dtype)
    bh = self.h0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_layers)
    bc = self.c0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_layers)
    for t_i in range(seq_len):
      bh, bc = self.pre_buffer(emb_rev[t_i], (bh, bc)) # (batch_size, self.hid_dim, self.num_layers)
      buffer_hiddens[t_i, :, :] = bh.contiguous().view(batch_size, -1) # (batch_size, self.hid_dim * self.num_layers)
      buffer_cells[t_i, :, :] = bc.contiguous().view(batch_size, -1) # (batch_size, self.hid_dim * self.num_layers)

    self.buffer.build_stack(batch_size, seq_len, buffer_hiddens[:, :, self.hid_dim * (self.num_layers - 1):], tokens_mask, self.gpuid)
    self.token_stack.build_stack(batch_size, self.stack_size, gpuid=self.gpuid)
    self.token_buffer.build_stack(batch_size, seq_len, emb_rev, tokens_mask, self.gpuid)
    self.stack(self.root_input.unsqueeze(0).expand(batch_size, self.input_dim), torch.ones(batch_size).type(self.long_dtype)) # push root

    buffer_hiddens = torch.cat([self.h0.expand(batch_size, 2, self.hid_dim).contiguous().view(batch_size, self.hid_dim * 2).unsqueeze(0), buffer_hiddens], dim=0)  # prepare for the indexing at end
    buffer_cells = torch.cat([self.c0.expand(batch_size, 2, self.hid_dim).contiguous().view(batch_size, self.hid_dim * 2).unsqueeze(0), buffer_cells], dim=0)  # prepare for the indexing at end

    stack_state, _ = self.stack.head() # (batch_size, hid_dim, num_layers)
    buffer_state = self.buffer.head() # (batch_size, hid_dim)
    stack_state = stack_state[:, :, -1] # (batch_size, hid_dim)

    token_stack_state = self.token_stack.head()
    token_buffer_state = self.token_buffer.head()
    stack_input =  token_buffer_state # (batch_size, input_size)
    action_state = self.h0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_layers) # (batch_size, hid_dim, num_lstm_layers)
    action_cell = self.c0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_layers) # (batch_size, hid_dim, num_lstm_layers)
    prev_action_state = action_state.clone()  # same as above, but sampled with argmax instead of averaged
    prev_action_cell = action_cell.clone()  # same as above, but sampled with argmax instead of averaged
    action_state = action_state[:, :, -1]

    # summary state memory bank which will be returned as "encoding"
    encodings = []

    # main loop
    outputs = torch.zeros((2 * seq_len, batch_size, len(self.actions))).type(self.dtype)
    batch_indexes = torch.arange(0, batch_size).type(self.long_dtype)
    step_length = 2 * seq_len
    step_i = 0
    # during decoding, some instances in the batch may terminate earlier than others
    # and we cannot terminate the decoding because one instance terminated
    while step_i < step_length:

      # get action decisions
      summary = self.summarize_states(torch.cat((stack_state, buffer_state, action_state), dim=1)) # (batch_size, self.state_dim)
      encodings.append(summary)

      # Gumbel softmax happens here
      action_dist = onmt.modules.utils.rand.gumbel_softmax_sample(self.state_to_actions(summary), self.gumbel_temperature, self.gumbel_beta) # (batch_size, len(actions))
      # ction_dist = self.softmax(self.state_to_actions(summary)) # (batch_size, len(actions))
      outputs[step_i, :, :] = action_dist.clone()

      # get rid of forbidden actions (only for decoding)
      forbidden_actions = self.get_valid_actions(batch_size) ^ 1
      num_forbidden_actions = torch.sum(forbidden_actions, dim=1)  # (batch_size,)
      eos = (num_forbidden_actions == (len(self.actions) - 1))  # (batch_size,)
      # if all actions except <pad> are forbidden for all sentences in the batch (end-of-sequence),
      # there is no sense continue decoding.
      if eos.all():
        break
      # otherwise, <pad> should only be allowed at the end of sequence
      else:
        forbidden_actions[:, -1] = (eos ^ 1)  
      # if an action is forbidden, assign a very small probability
      filtered_action_dist = action_dist.clone().masked_fill(forbidden_actions, -999)

      # Straigh-Through sample happens here
      _, action_i = torch.max(filtered_action_dist, dim=1) # (batch_size,)
      action_i.detach_()

      # translate action into stack & buffer ops
      # stack_op, buffer_op = self.map_action(action_i.data)
      stack_op = self.stack_action_mapping.index_select(0, action_i)
      buffer_op = self.buffer_action_mapping.index_select(0, action_i)

      # update stack, buffer and action state
      # note that to accomodate Gumbel Softmax sampling in the backward pass we are retuning all possible state outputs
      stack_state, _ = self.stack(stack_input, stack_op) # (3, batch_size, hidden_dim)
      buffer_state = self.buffer(stack_state[0], buffer_op) # (2, batch_size, hidden_dim) XXX: doesn't matter which input do we feed

      token_stack_state = self.token_stack(stack_input, stack_op) # (2, batch_size, hidden_dim)
      token_buffer_state = self.token_buffer(stack_input, buffer_op) # (2, batch_size, hidden_dim)

      stack_state_expanded = stack_state.index_select(0, self.stack_action_mapping + 1).permute(1, 2, 0)  # (batch_size, hidden_dim, |A|)
      buffer_state_expanded = buffer_state.index_select(0, self.buffer_action_mapping + 1).permute(1, 2, 0)  # (batch_size, hidden_dim, |A|)
      # uncomment token stack/buffer lines should token highway connection needs to be brought back
      # token_stack_state_expanded = token_stack_state.index_select(dim=0, self.stack_action_mapping).permute(1, 2, 0)  # (batch_size, hidden_dim, |A|)
      # token_buffer_state_expanded = token_buffer_state.index_select(dim=0, self.buffer_action_mapping).permute(1, 2, 0)  # (batch_size, hidden_dim, |A|)

      # action_input = self.action_emb(action_i) # (batch_size, action_emb_dim)
      prev_action_state = prev_action_state.unsqueeze(0).expand(len(self.actions), -1, -1, -1).reshape(-1, self.hid_dim, self.num_layers)
      prev_action_cell = prev_action_cell.unsqueeze(0).expand(len(self.actions), -1, -1, -1).reshape(-1, self.hid_dim, self.num_layers)
      action_state, action_cell = self.history(self.action_emb.weight.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, self.action_emb_dim), (prev_action_state, prev_action_cell))  # (batch_size * |A|, hid_dim, num_lstm_layers)
      prev_action_state = action_state.view(len(self.actions), batch_size, self.hid_dim, self.num_layers)[action_i, batch_indexes, :, :]  # (batch_size, hid_dim, num_lstm_layers)
      prev_action_cell = action_cell.view(len(self.actions), batch_size, self.hid_dim, self.num_layers)[action_i, batch_indexes, :, :]  # (batch_size, hid_dim, num_lstm_layers)

      action_state_expanded = action_state[:, :, -1].view(batch_size, len(self.actions), self.hid_dim)  # (batch_size, |A|, hid_dim)
      action_state_expanded = action_state_expanded.transpose(1, 2)  # (batch_size, hid_dim, |A|)

      # Averaging over p_G for backward pass happens here
      stack_state = torch.bmm(stack_state_expanded, action_dist.exp().unsqueeze(2)).squeeze()  # (batch_size, hidden_dim)
      buffer_state = torch.bmm(buffer_state_expanded, action_dist.exp().unsqueeze(2)).squeeze()  # (batch_size, hidden_dim)
      action_state = torch.bmm(action_state_expanded, action_dist.exp().unsqueeze(2)).squeeze().view(batch_size, self.hid_dim)  # (batch_size, hidden_dim)
      # token_stack_state = torch.bmm(token_stack_state_expanded, action_dist.unsqueeze(2)).squeeze()  # (batch_size, hidden_dim)
      # token_buffer_state = torch.bmm(token_buffer_state_expanded, action_dist.unsqueeze(2)).squeeze()  # (batch_size, hidden_dim)

      stack_input = self.token_buffer.head()

      step_i += 1

    # compute encoder_final
    final_stack_state, final_stack_cell = self.stack.head()  # (batch_size, hid_dim, num_lstm_layers)
    final_stack_state = final_stack_state.transpose(1, 2).reshape(-1, self.hid_dim)  # (batch_size * num_lstm_layers, hid_dim)
    final_stack_cell = final_stack_cell.transpose(1, 2).reshape(-1, self.hid_dim)  # (batch_size * num_lstm_layers, hid_dim)
    final_action_state, final_action_cell = prev_action_state.transpose(1, 2).reshape(-1, self.hid_dim), prev_action_cell.transpose(1, 2).reshape(-1, self.hid_dim)  # (batch_size * num_lstm_layers, hid_dim)
    final_buffer_state = buffer_hiddens[self.buffer.pos, batch_indexes, :].reshape(batch_size, self.hid_dim, self.num_layers)
    final_buffer_state = final_buffer_state.transpose(1, 2).reshape(-1, self.hid_dim)  # (batch_size * num_lstm_layers, hid_dim)
    final_buffer_cell = buffer_cells[self.buffer.pos, batch_indexes, :].reshape(batch_size, self.hid_dim, self.num_layers)
    final_buffer_cell = final_buffer_cell.transpose(1, 2).reshape(-1, self.hid_dim)  # (batch_size * num_lstm_layers, hid_dim)
    final_state = self.summarize_states(torch.stack([final_stack_state, final_buffer_state, final_action_state], dim=1).view(batch_size, self.num_layers, -1)).transpose(0, 1)  # (num_lstm_layers, batch_size, state_dim)
    final_cell = self.summarize_states(torch.stack([final_stack_cell, final_buffer_cell, final_action_cell], dim=1).view(batch_size, self.num_layers, -1)).transpose(0, 1)  # (num_lstm_layers, batch_size, state_dim)

    encoder_final = (final_state, final_cell)
    if self.use_bridge:
        encoder_final = self._bridge(encoder_final)

    # FIXME: outputs is not returned here, but at some point we'll have to print this in order to examine the trees
    return encoder_final, torch.stack(encodings, dim=0)


  def _initialize_bridge(self, rnn_type,
                         hidden_size,
                         num_layers):

    # LSTM has hidden and cell state, other only one
    number_of_states = 2 if rnn_type == "LSTM" else 1
    # Total number of states
    self.total_hidden_dim = hidden_size * num_layers

    # Build a linear layer for each
    self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                           self.total_hidden_dim,
                                           bias=True)
                                 for i in range(number_of_states)])


  def _bridge(self, hidden):
    """
    Forward hidden state through bridge
    """
    def bottle_hidden(linear, states):
      """
      Transform from 3D to 2D, apply linear and return initial size
      """
      size = states.size()
      result = linear(states.view(-1, self.total_hidden_dim))
      return F.relu(result).view(size)

    if isinstance(hidden, tuple):  # LSTM
      outs = tuple([bottle_hidden(layer, hidden[ix])
                    for ix, layer in enumerate(self.bridge)])
    else:
      outs = bottle_hidden(self.bridge[0], hidden)
    return outs


  def get_valid_actions(self, batch_size):
    action_mask = torch.ones(batch_size, len(self.actions)).type(self.long_dtype).byte()  # (batch_size, len(actions))

    for idx, action_str in enumerate(self.actions):
      # general popping safety
      sa = self.stack_action_mapping[idx].repeat(batch_size)
      ba = self.buffer_action_mapping[idx].repeat(batch_size)
      stack_forbid = ((sa == -1) & (self.stack.pos <= 1))
      buffer_forbid = ((ba == -1) & (self.buffer.pos <= 0))
      action_mask[:, idx] = (action_mask[:, idx] & ((stack_forbid | buffer_forbid) ^ 1))

      # transition-system specific operation safety
      if self.transSys == TransitionSystems.AER:
        la_forbid = (((self.buffer.pos <= 0) | (self.stack.pos <= 1)) & action_str.startswith("Left-Arc"))
        ra_forbid = (((self.stack.pos <= 0) | (self.stack.pos <= 0)) & action_str.startswith("Right-Arc"))
        action_mask[:, idx] = (action_mask[:, idx] & ((la_forbid | ra_forbid) ^ 1))

      if self.transSys == TransitionSystems.AH:
        la_forbid = (((self.buffer.pos <= 0) | (self.stack.pos <= 1)) & action_str.startswith("Left-Arc"))
        ra_forbid = (self.stack.pos <= 1) & action_str.startswith("Right-Arc")
        action_mask[:, idx] = (action_mask[:, idx] & ((la_forbid | ra_forbid) ^ 1))

    return action_mask


  def set_action_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]

      # If the action is unknown, leave the stack and buffer state as-is
      if self.transSys == TransitionSystems.AER:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AER_map.get(transition, (0, 0))
      elif self.transSys == TransitionSystems.AH:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AH_map.get(transition, (0, 0))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError


  def set_hard_composition_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]

      # If the action is unknown, read/write attention weights should all be 0
      if self.transSys == TransitionSystems.AER:
        self.composition_attn_h[idx], self.composition_attn_d[idx], self.composition_attn_b[idx] = \
          AER_hard_composition_map.get(transition, ([0] * self.composition_k * 2, [0] * self.composition_k * 2, [0] * self.composition_k * 2))
      if self.transSys == TransitionSystems.AH:
        self.composition_attn_h[idx], self.composition_attn_d[idx], self.composition_attn_b[idx] = \
          AH_hard_composition_map.get(transition, ([0] * self.composition_k * 2, [0] * self.composition_k * 2, [0] * self.composition_k * 2))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError

