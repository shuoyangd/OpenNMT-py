#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import editdistance

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-p', action='store' , dest='predictions',  required = True)
    opt.add_argument('-r', action='store' , dest='references',  required = True)
    opt.add_argument('-m', action='store' , dest='phone_map',  required = True)
    options = opt.parse_args()
    wer_per_line = []
    phone_map = {}
    for line in open(options.phone_map, 'r').readlines():
        k,v = line.strip().split()
        v = v if v != 'None' else None
        phone_map[k] = v
    assert len(phone_map) == 61
    predictions = open(options.predictions, 'r').readlines()
    references = open(options.references, 'r').readlines()
    assert len(predictions) == len(references)

    for p,r in zip(predictions, references):
        p = [phone_map[i] for i in p.strip().split()]
        p = [i for i in p if i is not None]
        r = [phone_map[i] for i in r.strip().split()[1:]] #REMOVE UTTERENCE ID!!
        r = [i for i in r if i is not None]
        #print(p)
        #print(r)
        e = editdistance.eval(p, r)
        wer = float(e) /float(len(r))
        wer_per_line.append(wer)
    ave_wer = sum(wer_per_line) / float(len(wer_per_line))
    print('wer:',ave_wer)
