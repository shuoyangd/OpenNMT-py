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
    phone_map = {}
    for line in open('./phones.61-39.map', 'r').readlines():
        k,v = line.strip().split()
        v = v if v != 'None' else ''
        phone_map[k] = v
    assert len(phone_map) == 61
    predictions = open(options.predictions, 'r').readlines()
    references = open(options.references, 'r').readlines()
    assert len(predictions) == len(references)
    for p,r in zip(predictions, references):
        p = [phone_map[i] for i in p.strip().split()]
        r = [phone_map[i] for i in r.strip().split()]
        e = editdistance.eval(p, r)
        wer = float(e) /float(len(r))
        wer_per_line.append(wer)
    ave_wer = sum(wer_per_line) / float(len(wer_per_line))
    print('wer:',ave_wer)
