#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import editdistance

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-p', action='store' , dest='predictions',  required = True)
    opt.add_argument('-r', action='store' , dest='references',  required = True)
    options = opt.parse_args()
    wer_per_line = []
    predictions = open(options.predictions, 'r').readlines()
    references = open(options.references, 'r').readlines()
    assert len(predictions) == len(references)
    for p,r in zip(predictions, references):
        p = p.replace('@@ ', '')
        r = r.replace('@@ ', '')
        p = p.strip().split()
        r = r.strip().split()[1:] #FIRST TOK is the Utterence ID
        e = editdistance.eval(p, r)
        wer = float(e) /float(len(r))
        wer_per_line.append(wer)
    ave_wer = sum(wer_per_line) / float(len(wer_per_line))
    print('wer:',ave_wer)
