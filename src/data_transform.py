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
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Need a work number and names of input!!!")
        print('For example, "./data_transform.py input.a.txt --with_graph"')
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
        fout = ('graph/')+ fout + ('transform.distribution.png')
        print('output graph is \"',fout,"\"")
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

    set min and max

    '''
    data_max = np.max(data,axis=0)
    data_min = np.min(data,axis=0)

'''

    Parameter Part and Global variable

'''
#super global:
rtn_data = []
#parameter:
operation_array = parameter.operation_array

#global:
recover_std = {}
recover_mean = {}
recover_max = {}
add_dim = {}
new_index = {}

'''

    API Region : Used by other code file

'''
def load_data(data2):    
    global data_min,data_max,data
    data = data2
    data_max = np.max(data,axis=0)
    data_min = np.min(data,axis=0)
    
def recover(transform_type,index):
    print('(anti-type):',index,transform_type)
    array = data[:,index]

    if transform_type =='log':
        data_max[index] = np.power(10.,data_max[index])
        data_min[index] = np.power(10.,data_min[index])
        data[:,index] = np.power(10.,array)

    if transform_type == 'none':
        data[:,index] = array

    if transform_type == 'normalise':       
        data[:,index] =(array * np.repeat([recover_std[index]],array.shape[0],axis=0) ) \
                + ( np.repeat([recover_mean[index]],array.shape[0],axis=0) )
    if transform_type == 'absmax_to_five':
        data[:,index] = array*recover_max[index]/5.

    if transform_type == 'negative_exp':
        data[:,index] = np.log10(array) * (-1)
    
    if transform_type == 'exp':
        data[:,index] = np.log10(array)

    if transform_type == 'log_negative_2_dim':
        data[:,index] == np.sign(add_dim[index])*np.power(10,array)

def get_recover_data(recover_length):

    for index in range(0,recover_length):
                
        for type_index in range(len(operation_array[index]) -1 ,-1, -1):            
            recover(operation_array[index][type_index],index)

    return data

def get_data():
    return np.transpose(rtn_data)
def get_new_index():
    return new_index

'''

    main function

'''
#main code
def transform(transform_type,index):    
    global data
    array = data[:,index]
    array_min = data_max[index]
    array_max = data_min[index]

    if transform_type =='log':
        print('(type):',index,transform_type)
        array = np.log10( array *(array>0.) + np.full((len(array),),1e-30) )        
        data_max[index] = np.log10(array_min)
        data_min[index] = np.log10(array_max)
            
        data[:,index] = array

    if transform_type == 'none':
        print('(type):',index,transform_type)
        data[:,index] = array

    if transform_type == 'normalise': 
        print('(type):',index,transform_type)       
        recover_std[index] = np.std(array,axis=0)
        recover_mean[index] = np.mean(array,axis=0)
        array= (array - np.repeat([recover_mean[index]],array.shape[0],axis=0) ) \
                / ( np.repeat([recover_std[index]+1e-30],array.shape[0],axis=0) )
        data[:,index] = array

    if transform_type == 'absmax_to_five':
        print('(type):',index,transform_type)
        recover_max[index] = max( np.abs(array_max) ,np.abs(array_min) )
        array = array / recover_max[index]*5.
        data[:,index] = array

    if transform_type == 'negative_exp':
        print('(type):',index,transform_type)
        array = np.power(10,array * (-1))
        data[:,index] = array
    
    if transform_type == 'exp':
        print('(type):',index,transform_type)
        array = np.power(10,array)
        data[:,index] = array
    
    if transform_type == 'log_negative_2_dim':
        print('(type):',index,transform_type)
        print('(add dim)')       
               
        data[:,index] = np.log10(np.abs(array))
        add_dim[index] = ( np.tanh(array) )

'''

    main part : tranfom the data

'''
#main operation
def main():
    print("========== transform ============")
    for index in range(0,data.shape[1]):
                
        for t_type_index in range(0,len(operation_array[index])):            
            transform(operation_array[index][t_type_index],index)        
            
        if index in recover_mean:
            print('recover_mean',index,recover_mean[index])
        if index in recover_std:
            print('recover_std',index,recover_std[index])
        if index in recover_max:
            print('recover_max',index,recover_max[index])

    for index in range(0,data.shape[1]):
        new_index[index] = len(rtn_data)
        print('new dim','from:'+str(index),'to:'+str(len(rtn_data)))
        rtn_data.append(data[:,index])            
        if index in add_dim:
            print('new dim','from:add_dim','to:'+str(len(rtn_data)))
            rtn_data.append(add_dim[index])
    print("=================================")

## if main additional part

def Plot(index_array):
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
        newindex = 'new_dim'
        for key in new_index:
            if new_index[key] == i:
                newindex = str(key)
        plt.title('variable:'+newindex)        
        plt.xlabel('variable:'+newindex+' (unit:std)')
        plt.ylabel('dot number')
        print('index:',i)
        bins = np.linspace(-6.,6.,100)
        plt.hist(rtn_data[int(index_array[i])], color='skyblue' ,bins=bins)
    plt.tight_layout()
    plt.savefig(fout)
    plt.show()

if __name__ == '__main__':
    main()    
    if with_graph:
        print('Show picture of tranform')
        Plot(np.array(range(0,len(rtn_data)))) ##transpose    