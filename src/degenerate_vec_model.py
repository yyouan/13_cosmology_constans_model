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
import degenerate_vec_model_train as model_train
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
    print('For example, "./degenerate_vec_model.py input.a.txt --name test1 --with_graph"')
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

training = parameter.degenerate_vec_training
training['Data'] = data[int(np.floor(data.shape[0]*training['Region'])) : int(np.floor(data.shape[0]*training['Region']) + np.floor(data.shape[0]*training['Rate']))] 
training['BatchSize'] = int(np.floor(training['Data'].shape[0]))
inputLen = parameter.degenerate_vec_inputLen

##input
new_index = data_transform.get_new_index()
inputLen = new_index[inputLen-1]+1
input_array = training['Data'][:,:(inputLen)]

##output
output_index_array = np.array(range(inputLen,data.shape[1]))#[ (65-1) , (21-1)] # 65.relic 67.DM mass
output_array = training['Data'][:, output_index_array]

print("input_array:",end='')
print(input_array)
print(input_array.shape)    
print("output_array:",end='')
print(output_array)
print(output_array.shape)

'''

    API Region

'''
##helper function
def get_input_array():
    return input_array

def get_output_array():
    return output_array

def Plot_output(y_array,z_array):
        import matplotlib.pyplot as plt
        plt.plot(y_array[:,0],y_array[:,1],"ko")
        plt.plot(z_array[:,0],z_array[:,1],"ro")
        plt.title('{relic to m_h}')
        plt.xlabel('relic')
        plt.ylabel('m_h')    
        plt.legend(['theory','model'],loc = 'upper left')
        plt.savefig("graph/"+name+'.degenerate_vec.model_output.png')
        plt.show()
        

def Plot_input(theory_array,model_array):

        import matplotlib.pyplot as plt
        plt.figure()
        
        if len(theory_array) == len(model_array) and theory_array.shape[1]== model_array.shape[1]:
            length = theory_array.shape[1]
            for i in range( 0 ,length):            
                plt.subplot( int(length/2)+1, 2, i+1 )
                plt.title('variable:'+str(i))
                plt.xlabel('variable:'+str(i)+' (unit:std)')
                plt.ylabel('dot number')    
                plt.legend(['theory(pink)','model(skyblue)'],loc = 'upper left')
                bins = np.linspace(-5.0, 5.0, 100)
                plt.hist(model_array[:,i], color='skyblue' ,bins=bins,histtype=u'step')
                plt.hist(theory_array[:,i], color='pink',bins=bins,histtype=u'step')
                                 

                '''
                plt.subplot( length, 2, (2*i+2) )
                plt.title('variable:'+str(i))
                plt.xlabel('variable:'+str(i)+' (unit:std)')
                plt.ylabel('dot number')    
                plt.legend(['theory(pink)','model(skyblue)'],loc = 'upper left')
                '''
                #print(len(theory_array[:,i]),len(model_array[:,i]))
                
        plt.savefig("graph/"+name+'.degenerate_vec.model_input.png')
        plt.show()
'''

    main part

'''
#model
model = model_train.model_train(name,input_array,output_array,training,data,with_graph)
#show and save result
print('=============show result by text==============')
training['std_prediction'] =  model.predict(input_array)
print("output",training['std_prediction'][0])
print("input",training['std_prediction'][1])
print('==============================================')

if with_graph == True:
    #show and save result    
    Plot_input(input_array,training['std_prediction'][0])
    Plot_output(output_array,training['std_prediction'][1])
