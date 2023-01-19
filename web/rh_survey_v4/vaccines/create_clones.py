#JRR Create clones for the vaccine file
import shutil
import os
import re
n_clones = 9
original_name = 'V1.lsg'
original_fname = os.path.join('.', original_name)
for i in range(2,n_clones + 1):
    n = str(i)
    clone_name = 'V' + n + '.lsg'
    clone_fname = os.path.join('.', clone_name)
    #shutil.copy(original_fname, clone_fname)
    with open(original_fname, 'rt') as fin:
        with open(clone_fname, 'wt') as fout:
            for line in fin:
                new_line = line.replace('#1', '#' + n)
                new_line = new_line.replace('AO01', 'AO0' + n)
                fout.write(new_line)


