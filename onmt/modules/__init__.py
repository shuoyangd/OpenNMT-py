from nmt.onmt.modules.UtilClass import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from nmt.onmt.modules.Gate import ContextGateFactory
from nmt.onmt.modules.GlobalAttention import GlobalAttention
from nmt.onmt.modules.ConvMultiStepAttention import ConvMultiStepAttention
from nmt.onmt.modules.ImageEncoder import ImageEncoder
from nmt.onmt.modules.CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from nmt.onmt.modules.StructuredAttention import MatrixTree
from nmt.onmt.modules.Transformer import TransformerEncoder, TransformerDecoder
from nmt.onmt.modules.Conv2Conv import CNNEncoder, CNNDecoder
from nmt.onmt.modules.MultiHeadedAttn import MultiHeadedAttention
from nmt.onmt.modules.StackedRNN import StackedLSTM, StackedGRU
from nmt.onmt.modules.Embeddings import Embeddings
from nmt.onmt.modules.WeightNorm import WeightNormConv2d

from nmt.onmt.modules.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from nmt.onmt.modules.SRU import SRU


# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU, ContextGateFactory,
           CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])
