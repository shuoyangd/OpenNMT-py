#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('-r', action='store' , dest='ref', required = True)
    opt.add_argument('-d', action='store' , dest='data_type', required = True)
    options = opt.parse_args()
    ref_formated = open(options.ref + '.formatted', 'w')
    for ref_line in open(options.ref, 'r').readlines(): 
        idx, _ref_line = ref_line.strip().split(None, 1)
        if options.data_type == 'wsjchars':
            spkr_idx = '(' + idx[:3] + '-'  + idx + ')'
        elif options.data_type == 'chime4':
            spkr, u_id, noise, real = idx.split('_')
            #spkr_idx = '(' + noise + real + '-'  + idx + ')'
            spkr_idx = '(' + idx[:3] + '-'  + idx + ')'
        elif options.data_type == 'st':
            file_id,caller_AB, st,end = idx.split('-')
            spkr_idx = '(' + file_id + caller_AB + '-' + file_id+caller_AB+st+end + ')'
        else:
            raise BaseException("unknown data_type")
        ref_formated.write(_ref_line + ' ' + spkr_idx + '\n')
    ref_formated.flush()
    ref_formated.close()
