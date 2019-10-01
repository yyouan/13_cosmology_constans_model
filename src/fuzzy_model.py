from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks
import fuzzy_model_train as fuzzy_model_train
import fuzzy_fuzzy_function as fuzzy
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import parameter as parameter
import data_transform as data_transform

np.random.seed(10)

#interface variable
name="" 
with_graph = False

'''

    Sub-Command Part

'''
print('================== In creating model =====================')
if len(sys.argv) < 4:
    print("Need a work number and names of input!!!")
    print('For example, "./data_transform.py input.a.txt --name test1 --with_graph"')
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
    name = sys.argv[3]       
    fout = ('graph/')+ fout + ('training_result.name.')+ name +('.distribution.png')
    print('output graph is \"',fout,"\"")
    data_type = sys.argv[1].split('.')[-1]
    
if len(sys.argv) == 5:
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

    Transform Data in data_transform.py

'''
data_transform.load_data(data)
data_transform.main()
data = data_transform.get_data()

'''

    Parameter Part and Global variable

'''
#declare variables

training = parameter.fuzzy_training
training['Data'] = data[int(np.floor(data.shape[0]*training['Region'])) : int(np.floor(data.shape[0]*training['Region']) + np.floor(data.shape[0]*training['Rate']))] 
training['BatchSize'] = int(np.floor(training['Data'].shape[0]))
inputLen = parameter.fuzzy_inputLen

##input
new_index = data_transform.get_new_index()
inputLen = new_index[inputLen-1]+1
input_array = training['Data'][:,:(inputLen)]

output_array = input_array

##output
output_index_array = np.array(range(inputLen,data.shape[1]))#[ (65-1) , (21-1)] # 65.relic 67.DM mass
input_array = training['Data'][:, output_index_array]

print("input_array:",end='')
print(input_array)
print(input_array.shape)    
print("output_array:",end='')
print(output_array)
print(output_array.shape)

'''

    Fuzzilize

'''
##fuzzilize:
training['Output_Partion_Fuzzy'] =  fuzzy.fuzzy(input_array,rrange=0.5,partion=training['Output_Partion'])
input_array = training['Output_Partion_Fuzzy'].cut()
training['Input_Partion_Fuzzy'] = fuzzy.fuzzy(output_array,rrange=0.5,partion=training['Input_Partion'])
output_array = training['Input_Partion_Fuzzy'].cut()

'''

    API Region

'''
##helper function
def get_input_array():
    return input_array

def get_output_array():
    return output_array

def fuzzyPlot(theory_array,model_array):

        import matplotlib.pyplot as plt
        plt.figure()
        
        if len(theory_array) == len(model_array) and theory_array.shape[1]== model_array.shape[1]:
            length = theory_array.shape[1]

            for i in range( 0 , length):                

                theory_total = np.sum(theory_array[:,i,:] ,axis=0)

                model_total = np.sum(model_array[:,i,:] ,axis=0) /(1.0 + np.max(model_array[:,i,:] ,axis=0)) * np.max(theory_array[:,i,:] ,axis=0)
                ncols=0
                if length%2 == 0:
                    ncols = length/2
                else:
                    ncols = length/2+1
                plt.subplot( ncols, 2, i+1 )
                plt.title('variable:'+str(i))
                plt.xlabel('variable:'+str(i)+' (unit:std)')
                plt.ylabel('dot number')    
                plt.legend(['theory(red)','model(blue)'],loc = 'upper left')
                bins = training['Input_Partion_Fuzzy'].fuzzy_linspace()
                plt.plot(bins, theory_total, 'r--')
                plt.plot(bins, model_total, 'b--')
                
                
        plt.savefig("graph/"+name+'.fuzzy.model_input.png')
        plt.show()
'''

    main part

'''
#model
model = fuzzy_model_train.fuzzyload(name,input_array,output_array,training,data,output_index_array,Output_partion=training['Input_Partion'],Input_partion=training['Output_Partion'],with_graph=with_graph)
#show and save result
print('=============show result by text==============')
print('input_array')
print(input_array)
print('predict_array')      
print(model.predict(input_array))
print('==============================================')
if with_graph == True:
    fuzzyPlot(output_array,model.predict(input_array))
