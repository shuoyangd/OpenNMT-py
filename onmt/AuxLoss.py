from nmt.onmt.Loss import NMTLossCompute

class AuxLossCompute(NMTLossCompute):

    def __init__(self, generator, tgt_vocab):
        super(AuxLossCompute, self).__init__(generator, tgt_vocab)


    def make_shard_state(self, batch, output, range_, attns=None):
        """ See base class for args description. """
        return {
            "output": output,
            "target": batch.aux_tgt[range_[0] + 1: range_[1]],
        }
