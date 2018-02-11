#!/usr/bin/env python
import argparse
import os
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
    for pf in aug2id:
        src_in = open(os.path.join(options.folder, pf + '.src'))
        tgt_in = open(os.path.join(options.folder, pf + '.src'))
        aug_idx = aug2id[pf]
        for line in src_in:
            toks = line.strip().split()
            decorated_toks = [ tok + "_" + str(aug_idx) for tok in toks ]
            src_out.write(" ".join(decorated_toks) + "\n")
        for line in tgt_in:
            tgt_out.write(pf + " " + line)
    src_out.flush()
    src_out.close()
    tgt_out.flush()
    tgt_out.close()
