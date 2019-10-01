from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import sys

fout=""

'''

    API Part

'''
def file_name(name):
    global fout
    fout = ""
    raw_fout = (name)
    if len(raw_fout.split('/')) > 1:
        raw_fout = raw_fout.split('/')[-1]
    elif len(raw_fout.split('\\')) > 1:
        raw_fout = raw_fout.split('\\')[-1]
    for string in raw_fout.split('.')[:-1]:        
        if string!='':        
            fout += (string+'.')
    print('input file is \"', sys.argv[1],"\"")
    fout = ('graph/')+ fout + ('distribution.png')

'''

    main function

'''
def Plot(plot_data,index_array):

    import matplotlib.pyplot as plt    
    length = len(index_array)
    plt.figure(dpi = 50,figsize=(80,80))
    for i in range( 0 , length):
        ncols=0
        if length%2 == 0:
            ncols = length/2
        else:
            ncols = length/2+1
        plt.subplot( ncols, 2, i+1 )
        plt.title('variable:'+str(i))
        plt.xlabel('variable:'+str(i)+' (unit:std)')
        plt.ylabel('dot number')
        bins = np.linspace(np.min(plot_data[:,index_array[i]]),np.max(plot_data[:,index_array[i]]),100)
        plt.hist(plot_data[:,index_array[i]], color='skyblue' ,bins=bins)
    plt.tight_layout()
    plt.savefig(fout)
    plt.show()

'''
    
    main part

'''

if(__name__ == '__main__'):
    
    '''

    Sub-Command Part

    '''    

    if len(sys.argv) < 2:
        print("Need a work number and names of input!!!")
        print('For example, "./draw.py input.a.txt"')
        exit()
    elif sys.argv[1].split('.')[-1] != "dat" and sys.argv[1].split('.')[-1] != "txt" and sys.argv[1].split('.')[-1] != "csv":
        print("data type should be dat/txt/csv")
        exit()
    else:
        filename=sys.argv[1]
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
        fout = ('graph/')+ fout + ('distribution.png')
        print('output file is \"',fout,"\"")
        data_type = sys.argv[1].split('.')[-1]

    '''

        Read File

    '''

    #Loading data
    print("load...")
    if data_type == "txt":
        data = np.loadtxt(filename,dtype='float64')
    elif data_type == "dat":
        data = np.fromfile(filename, dtype='float64')
    elif data_type == "csv":
        data = np.genfromtxt(filename, delimiter=',', dtype='float64')    
    
    
    Plot(data,np.array(range(0,data.shape[1])))
    print('finish drawing')


