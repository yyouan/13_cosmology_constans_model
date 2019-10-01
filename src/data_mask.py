#!/usr/bin/python
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import parameter as parameter

with_graph = False

'''

    Sub-Command Part

'''
if len(sys.argv) < 2:
    print("Need a work number and names of input!!!")
    print('For example, "./data_mask.py input.a.txt --with_graph"')
    exit()
elif sys.argv[1].split('.')[-1] != "dat" and sys.argv[1].split('.')[-1] != "txt" and sys.argv[1].split('.')[-1] != "csv":
    print("data type should be dat/txt/csv")
    exit()
else:
    fname=sys.argv[1]
    fout=""
    raw_fout = (sys.argv[1])
    if len(raw_fout.split('/')) > 1:
        raw_fout = raw_fout.split('/')[-1]
    elif len(raw_fout.split('\\')) > 1:
        raw_fout = raw_fout.split('\\')[-1]
    for string in raw_fout.split('.')[:-1]:
        if string!='':        
            fout += (string+'.')
    print('input file is \"', sys.argv[1],"\"")
    fout += ('mask.txt')
    print('output file is \"',fout,"\"")
    data_type = sys.argv[1].split('.')[-1]    

if len(sys.argv) == 3:
    with_graph = True

'''

    Read File

'''
#Loading data
print("load...")
if data_type == "txt":
    data = np.loadtxt(fname,dtype='float64')
elif data_type == "dat":
    data = np.fromfile(fname, dtype='float64')
elif data_type == "csv":
    data = np.genfromtxt(fname, delimiter=',', dtype='float64')

'''

    parameter part

'''
#decare an array for the index
input_index_array = np.array(range(0,data.shape[1]))

# number of bin per dimension
mask_index = parameter.mask_index
print("mask index:", mask_index)

'''

     Main

'''

# log thinned data in file
'''input_mask = np.ones(inputLen, dtype=bool)
input_mask[[(1-1),(12-1),(13-1),(14-1),(17-1),(18-1),(19-1),(20-1)]] = False
raw_output_array = input_array[:,input_mask]'''
mask = np.ones(data.shape[1], dtype=bool)
mask[mask_index] = False
data = data[:,mask]

np.savetxt(fout,data)

#draw pitcture if argument is given
if(with_graph):
    import draw as draw

    draw.file_name(fname)
    print('Distribution','->draw to',draw.fout)
    draw.Plot(data,np.array(range(0,data.shape[1])))

