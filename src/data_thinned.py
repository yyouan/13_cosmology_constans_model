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
if len(sys.argv) < 3:
    print("Need a work number and names of input!!!")
    print('For example, "./data_thin.py input.a.txt 3 --with_graph"')
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
    fout += (sys.argv[2]+'.thinned.txt')
    print('output file is \"',fout,"\"")
    data_type = sys.argv[1].split('.')[-1]
    grid_partition = float(sys.argv[2])

if len(sys.argv) == 4:
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
crteria_index = parameter.crteria_index
criteria_medium = parameter.criteria_medium
print("criteria is index:",crteria_index," mean:",criteria_medium)

'''

    global variables

'''
# ???????
np.random.seed(10)

# To pick the max and min for each column
data_max = np.max(data,axis=0)
data_min = np.min(data,axis=0)
chosen_data= {}
chosen_data_num= {}
thinned_data = []

'''

    API Function

'''
# API region:

def get_thinned_data():
    global thinned_data
    return np.array(thinned_data)


'''

    Process Function

'''

def thin():
    global data,data_max,data_min,chosen_data,chosen_data_num,thinned_data
    print('Performing thinning...')
    print('Before bin,we have: ',data.shape[0])    
    
    for item in range(0,data.shape[0]): # run for all rows

        #print("check: "+str(item))
        #print('max',str(data_max))
        #print('min',str(data_min))


        '''
          Creating a key to store the bin ID of all dimensions.
          key is a string, renew for each row.

        '''

        key =""
        for index in input_index_array:  # run for all cols            
            key += str(int( (data[item,index] - data_min[index])/(1e-15 +data_max[index] - data_min[index]) *grid_partition ))  
            # find out the tag id for each dimension.
            key +=','


        # chosen_data is dicionary
        # chosen_data_num[key] is dicionary
        # If there is no key inside chosen_data then we create a element of dictionary.
        # If yes,
        if key in chosen_data:
            chosen_data_num[key] += 1

            # Based on the criteria to pick the best point        

            if np.abs(chosen_data[key][0][crteria_index] - criteria_medium) > np.abs(data[item,index] - criteria_medium):
                chosen_data[key][0] = data[item][:]

        else:            
            chosen_data[key] = [data[item,:]]
            chosen_data_num[key] = 1
'''

     Main


'''

thin()

# log thinned data in file
for key in chosen_data:
        for item in chosen_data[key]:
            thinned_data.append(item)
print('After bin, we have: ',len(chosen_data))

np.savetxt(fout,get_thinned_data())

#draw pitcture if argument is given
if(with_graph):
    import draw as draw

    draw.file_name(fname)
    print('Input','->draw to',draw.fout)
    draw.Plot(data,np.array(range(0,data.shape[1])))

    draw.file_name(fout)
    print('Output','->draw to',draw.fout)
    draw.Plot(get_thinned_data() ,np.array(range(0,get_thinned_data().shape[1])))

#show the data number in every region
'''print('(result:real_fat_size)')
print(chosen_data_num)'''

