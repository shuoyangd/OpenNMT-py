from datetime import datetime
from tensorboardX import SummaryWriter


def init(tensorboard_log_dir, comment="Onmt"):
  global _WRITER
  _WRITER = SummaryWriter(tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"), comment)
