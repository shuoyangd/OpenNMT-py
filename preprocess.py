# -*- coding: utf-8 -*-

import argparse
import codecs
import torch

import onmt
import onmt.IO
import opts

from collections import Counter
from torchtext.vocab import Vocab

parser = argparse.ArgumentParser(description='preprocess.py')
opts.add_md_help_argument(parser)


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-data_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-data_img_dir', default=".",
                    help="Location of source images")

parser.add_argument('-train_aug_src', required=True,
                    help="Path to the augmented source data")
parser.add_argument('-train_aug_tgt', required=True,
                    help="Path to the augmented target data")
parser.add_argument('-train_audio_tgt', required=True,
                    help="Path to the audio frame target data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opts.preprocess_opts(parser)

opt = parser.parse_args()
torch.manual_seed(opt.seed)


def main():
    print('Preparing training ...')
    num_aug_instances = 0
    with codecs.open(opt.train_aug_src, "r", "utf-8") as aug_src_file:
        num_aug_instances += 1
        aug_src_line = aug_src_file.readline().strip().split()
        _, _, nFeatures = onmt.IO.extract_features(aug_src_line)

    audio_word_counter = Counter()   
    num_audio_instances = 0
    data_names = []
    with codecs.open(opt.train_audio_tgt, "r", "utf-8") as audio_tgt_file:
        num_audio_instances += 1
        audio_tgt_line = audio_tgt_file.readline().strip().split()
        data_names.append(audio_tgt_line[0])
        for tok in audio_tgt_line:
            audio_word_counter[tok] += 1
    audio_tgt_vocab = Vocab(audio_word_counter)

    mix_fac = num_aug_instances / (num_audio_instances + num_aug_instances)

    fields = onmt.IO.ONMTDataset.get_fields(nFeatures)
    print("Building training...")
    train = onmt.IO.ONMTDataset(opt.train_aug_src, opt.train_aug_tgt, fields, opt)
    train.mix_fac = mix_fac
    train.data_names = data_names

    print("Building vocab...")
    onmt.IO.ONMTDataset.build_vocab(train, opt)
    train.fields["tgt"].vocab = onmt.IO.merge_vocabs(train.fields["tgt"].vocab, audio_tgt_vocab)

    print("Building valid...")
    valid = onmt.IO.ONMTDataset(opt.valid_tgt, opt.valid_tgt, fields, opt)
    print("Saving train/valid/fields")

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.ONMTDataset.save_vocab(fields),
               open(opt.save_data + '.vocab.pt', 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))


if __name__ == "__main__":
    main()
