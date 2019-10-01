import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense,Input,concatenate
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

np.random.seed(10)

#interface variable
name="" 
with_graph = False
from_data = 0 
to_data = 1

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
    dout = fout + ('training_result.name')+ name +('.distribution.txt')       
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
raw_input_array = data[from_data:to_data,:(inputLen)]
recover_length = data.shape[1]

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

##output
output_index_array = np.array(range(new_inputLen,data.shape[1]))#[ (65-1) , (21-1)] # 65.relic 67.DM mass
output_array = data[:, output_index_array]

## compute degenerate vector's dimension
degenerate_size = int(np.abs( input_array.shape[1] - output_array.shape[1] ))

def Plot_array_2dim(y_array):
    import matplotlib.pyplot as plt
    plt.plot(y_array[:,0],y_array[:,1],"ko")
    plt.title('{relic to m_h}')
    plt.xlabel('relic')
    plt.ylabel('m_h')
    plt.savefig(name+'use_rrbm_output')    
    plt.show()
    

def Plot_input(theory_array,model_array):

        import matplotlib.pyplot as plt
        plt.figure()
        
        if len(theory_array) == len(model_array) and theory_array.shape[1]== model_array.shape[1]:
            length = theory_array.shape[1]
            for i in range( 0 ,length):
                print('length',length,i)            
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
                
        plt.savefig("graph/"+name+'.degenerate_vec.prediction_input.png')
        plt.show()

'''

    Load model from json and load weights from h5

'''         
# load json and create model
json_file = open("log/" +name +'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights( "log/" +name + ".h5")
print("Loaded model from disk")

'''

    reconstruct the model

'''
#acuquire distribution of degenerate output:
forward_1 = loaded_model.get_layer(name='forward_1') (loaded_model.input)
forward_2 = loaded_model.get_layer(name='forward_2') (forward_1)
degenerate_layer = loaded_model.get_layer(name='degenerate') (forward_2)
pre_model = Model(inputs= loaded_model.input,outputs= degenerate_layer )
print(pre_model.summary())
degenerate_array = pre_model.predict(input_array[from_data:to_data,:])
degenerate_mean = np.mean(degenerate_array,axis=0)
degenerate_std = np.std(degenerate_array,axis=0)
print('degenerate:(mean,std):',end='')
print( degenerate_mean,degenerate_std )

# create sub_model to predict backward
output_layer = Input(shape=(output_array.shape[1],))
degenerate_layer = Input(shape=(degenerate_size,))
out = concatenate([output_layer,degenerate_layer])
backward_1 = loaded_model.get_layer(name='backward_1') (out)
backward_2 = loaded_model.get_layer(name='backward_2') (backward_1)
recover_layer = loaded_model.get_layer(name='recover') (backward_2)
sub_model = Model(inputs=[output_layer,degenerate_layer],outputs=recover_layer)

print(sub_model.summary())

'''

    Prediction and Show Prediction

'''
##time evalution:
import time
start_time = time.time()

#prediction:
Plot_array_2dim(output_array[from_data:to_data,:])
std_prediction = []

for index in range(from_data,to_data):
    for times in range(0,1):
        #notice it's batchlike operation
        std_prediction.append( 
            sub_model.predict( 
                [ [output_array[index,:]], 
                  [degenerate_array[np.random.randint(0,degenerate_array.shape[0])]]
                ]
            )[0] 
        )

std_prediction = np.array(std_prediction)
data[:,:(new_inputLen)] =  std_prediction
data_transform.load_data(data)
data = data_transform.get_recover_data(recover_length)
np.savetxt(dout,data)
if with_graph == True:
    Plot_input( raw_input_array, data[:,:(inputLen)])
