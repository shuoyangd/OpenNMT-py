import fnmatch
import os
import sys

# Data preparation script of the augmented training files
# Preparation done:
# 
# 1. For all source-side training files, the word tokens will be
#    decorated with the data source.
# 2. For all target-side training files, the file lines will start
#    with a tag denoting the data source.
#
# The data that to be prepared must follow the following naming
# coventions:
# 
# [prefix].aug.[phn|noise|trans|...].train.[src|tgt]
# 
# Note that audio frame files should not be prepared.
# 
# The output file after the data preparation will be:
#
# [prefix].aug.train.[src|tgt]

if len(sys.argv) < 2:
  sys.stderr.write("usage: python prepare_aug_data.py aug_file_prefix\n")
prefix_path = sys.argv[1]
prefix_dir = os.path.dirname(prefix_path)

src_out = open(prefix_path + ".aug.train.src", 'w')
tgt_out = open(prefix_path + ".aug.train.tgt", 'w')
for filename in os.listdir(prefix_dir):
  fn_prefix = os.path.basename(prefix_path)
  if fnmatch.fnmatch(filename, fn_prefix + "*.aug.train.*"):
    pass
  elif fnmatch.fnmatch(filename, fn_prefix + ".aug.*") and \
      fnmatch.fnmatch(filename, fn_prefix + "*.train.src"):
    print("processing {0}".format(filename))
    data_source = filename.split('.')[2].upper()
    src_in = open(os.path.join(prefix_dir, filename))
    for line in src_in:
      toks = line.strip().split()
      decorated_toks = [ tok + "_" + data_source for tok in toks ]
      src_out.write(" ".join(decorated_toks) + "\n")
  elif fnmatch.fnmatch(filename, fn_prefix + ".aug.*") and \
      fnmatch.fnmatch(filename, fn_prefix + "*.train.tgt"):
    print("processing {0}".format(filename))
    data_source = filename.split('.')[2].upper()
    tgt_in = open(os.path.join(prefix_dir, filename))
    for line in tgt_in:
      newtoks = line.split(None, 1)
      newtoks.insert(1, data_source)
      tgt_out.write(" ".join(newtoks))

src_out.close()
tgt_out.close()

