#!/usr/bin/env python
import argparse
import os
import random
from collections import namedtuple
#NEW USAGE:
#python prepare_aug_data.py --folder ./dummy_data/ --prefix_list dummy.noise dummy.phn --prefix_out dummy
START='<START>'
ReaderState = namedtuple('ReaderState', 'id name src_line, tgt_line, src_reader tgt_reader')

def get_remaining(aug_readers):
    cond = [rs for rs in aug_readers if rs.src_line.strip() != '']
    return cond 

def advance_all(aug_readers):
    return [advance(rs) for rs in aug_readers]

def advance(rs): 
    return ReaderState(rs.id, rs.name, rs.src_reader.readline(), rs.tgt_reader.readline(), rs.src_reader, rs.tgt_reader)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('--folder', dest='folder', required = True)
    opt.add_argument('--prefix_out', dest='prefix_out', required = True)
    opt.add_argument('--prefix_list', nargs='+', dest='prefix_list', required = True)
    options = opt.parse_args()
    src_out = open(os.path.join(options.folder, options.prefix_out + ".aug.train.src"), 'w')
    tgt_out = open(os.path.join(options.folder, options.prefix_out + ".aug.train.tgt"), 'w')
    aug2id = {k:i+1 for i,k in enumerate(options.prefix_list)}
    aug_readers = [ReaderState(idx_pf, pf, START , START, open(os.path.join(options.folder, pf + '.src'), 'r'), open(os.path.join(options.folder, pf + '.tgt'), 'r')) for idx_pf, pf in enumerate(options.prefix_list)]
    while len(aug_readers) > 0:
        random.shuffle(aug_readers)
        for rs in aug_readers:
            if rs.src_line == START:
                rs = advance(rs)
            else:
                pass
            toks = rs.src_line.strip().split()
            decorated_toks = [tok + "_" + str(rs.id) for tok in toks]
            src_out.write(" ".join(decorated_toks) + "\n")
            tgt_out.write(rs.name + " " + rs.tgt_line)
        aug_readers = advance_all(aug_readers)
        aug_readers = get_remaining(aug_readers)

    src_out.flush()
    src_out.close()
    tgt_out.flush()
    tgt_out.close()
