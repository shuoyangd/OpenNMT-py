#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import editdistance

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-p', action='store' , dest='prediction',  required = True)
    opt.add_argument('-r', action='store' , dest='reference',  required = True)
    options = opt.parse_args()
    wer_per_line = []
    predictions = open(options.predictions, 'r').readlines()
    references = open(options.reference, 'r').readlines()
    assert len(predictions) == len(references)
    for p,r in zip(predictions, references):
        p = p.strip().split()
        r = r.strip().split()
        e = editdistance.eval(p, r)
        wer = float(e) /float(len(r))
        wer_per_line.append(wer)
    print sum(wer_per_line) / float(len(wer_per_line))
