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
from keras.models import model_from_json

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import parameter as parameter
import data_transform as data_transform
import fuzzy_fuzzy_function as fuzzy

np.random.seed(10)

#interface variable
name="" 
with_graph = False
from_data = 0 
to_data = 1
training = parameter.fuzzy_training

'''

    Sub-Command Part

'''
print('================== In model predict =====================')
if len(sys.argv) < 10:
    print("Need a work number and names of input!!!")
    print('For example, "./data_transform.py input.a.txt --name test1 --inputLen 11 --from <the index test_data from> --to <the index test_data to> --with_graph --weight <weight-file>"')
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
    dout = fout + ('training_result.name')+ name +('.output_distribution.txt')       
    fout = ('graph/')+ fout + ('training_result.name')+ name +('.distribution.png')
    print('output graph is \"',fout,"\"")
    print('output file is \"',dout,"\"")
    data_type = sys.argv[1].split('.')[-1]
    inputLen = int( sys.argv[5] )
    from_data = int (sys.argv[7])
    to_data = int (sys.argv[9])

if len(sys.argv) == 11 or len(sys.argv) == 13:
    with_graph = True

if len(sys.argv) == 13: 
    weight = sys.argv[12]
elif len(sys.argv) == 12:
    weight = sys.argv[11]    
else:
    weight =""

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

## record input before transform

'''

    Transform Data in data_transform.py

'''
data = data[from_data:to_data,:]
data_transform.load_data(data)
data_transform.main()
data = data_transform.get_data()

##input
new_index = data_transform.get_new_index()
new_inputLen = new_index[inputLen-1]+1
input_array = data[:,:(new_inputLen)]

output_array = input_array

##output
output_index_array = np.array(range(new_inputLen,data.shape[1]))
input_array = data[:, output_index_array]

#fuzzilize
training['Output_Partion_Fuzzy'] =  fuzzy.fuzzy(input_array,rrange=0.5,partion=training['Output_Partion'])
input_array = training['Output_Partion_Fuzzy'].cut()
training['Input_Partion_Fuzzy'] = fuzzy.fuzzy(output_array,rrange=0.5,partion=training['Input_Partion'])
output_array = training['Input_Partion_Fuzzy'].cut()

'''

    Load model from json and load weights from h5

'''
# load json and create model
json_file = open( "log/"+ name +'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
if weight=="":
    loaded_model.load_weights( "log/"+ name + ".h5")
else:
    loaded_model.load_weights( weight )

print("Loaded model from disk")

'''

    Plot function

'''
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
                
                
        plt.savefig("graph/"+name+'.fuzzy.prediction_input.png')
        plt.show()

'''

    Prediction

'''
array_output =  loaded_model.predict(input_array)
np.savetxt(dout,data)
if with_graph == True:
    fuzzyPlot(output_array, array_output)
