#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
#(4k0-4k0c0301)
if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-r', action='store' , dest='ref', required = True)
    opt.add_argument('-o', action='store' , dest='out', required = True)
    opt.add_argument('-d', action='store' , dest='data_type',  required = True)
    options = opt.parse_args()
    out_formated = open(options.out + '.formatted', 'w')
    for ref_line, out_line in zip(open(options.ref, 'r').readlines(), open(options.out, 'r').readlines()):
        idx, _ref_line = ref_line.strip().split(None, 1)
        _out_line = out_line.strip()
        if options.data_type == 'wsjchars':
            spkr_idx = '(' + idx[:3] + '-'  + idx + ')'
        elif options.data_type == 'chime4':
            spkr, u_id, noise, real = idx.split('_')
            spkr_idx = '(' + noise + real + '-'  + idx + ')'
        else:
            raise BaseException("unknown data_type")
        out_formated.write(_out_line + ' ' + spkr_idx + '\n')
    out_formated.flush()
    out_formated.close()
