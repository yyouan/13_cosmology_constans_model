import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input,Flatten,Reshape
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks

'''

    main function : do all trainning here

'''
def fuzzyload( name,input_array,output_array,training,data,output_index_array,Output_partion,Input_partion,with_graph):
    
    print('================== In model trainning =====================')
    '''

    layer construction : we need to tune them

    '''
    model = Sequential()

    ## provide a normal distribution
    ## we need tune stddev for a good scaling
    normal = initializers.RandomNormal(mean=0.0, stddev=1.2, seed=10)
    print('input_size:',input_array.shape)
    print('output_size:',output_array.shape)
    
    '''DROP = Dropout(0.9)
    dense_3 = Dense(units = 100 , input_dim = inputLen , kernel_initializer= normal,
                    bias_initializer=normal , activation = 'relu')
    BN_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None) 
    INPUT = Input(shape=(input_array.shape[1],Input_partion))'''
    
    ## FLN is needed because now input is rank 2
    FLN = Flatten( input_shape=(input_array.shape[1],Input_partion))
    output_layer = Dense( units = 100 , kernel_initializer = normal, bias_initializer=normal , activation = 'relu') #only positive value
    dense_2 = Dense(units = 100, kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu')      
    dense_1 = Dense( units = output_array.shape[1]*Output_partion , kernel_initializer= normal,
                    activation = 'softplus')
    RSHP = Reshape((output_array.shape[1],Output_partion))
    layer_list = [FLN,output_layer,dense_2,dense_1,RSHP]   

 
    '''

    trainning

    '''

    ## create model
    for layer in layer_list:
        model.add(layer)
    print(model.summary())

    #showing the prediction of last 5 data for scaling adjusting
    print('==========showing the prediction of last 5 data==========')
    training['prediction'] =  model.predict(input_array[-5:-1,:]) 
    print(training['prediction'])
    print('=========================================================')

    ##compile model
    ## lr is learing rate : we need to tune it for good learning
    ## loss is loss function ,metric is only for show how good is the trainging result
    ## mse: mean squred error , mae :mean absolute error
    adam = optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    model.compile(optimizer=adam,
                loss= 'categorical_crossentropy',
                metrics=['categorical_accuracy'])

    ##time evalution:
    import time
    start_time = time.time()

    ##training mode
    ##Callbacks:
    # tensorboard viewing browser
    # checkpoint : save weight every period
    training['Callbacks'] = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    filepath = "log/" + name + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    training['Checkpoint'] = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    print('input',input_array.shape)
    print('output',output_array.shape)
    training['History'] = model.fit(input_array,output_array
                                ,validation_split = training['Validation_split']
                                ,epochs = training['Epochs']
                                ,batch_size=training['BatchSize'] ,verbose=1 , callbacks = []) #verbose for show training process
    
    print("trainning cost :: --- %s seconds ---" % (time.time() - start_time))

    '''

    save result of this trainning

    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open("log/"+ name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights( "log/"+ name+".h5")
    print("Saved model to disk")
    
    if with_graph == True:
        show_train_history(name,training['History'],'categorical_accuracy','val_categorical_accuracy')
     
    return model

def show_train_history(name,training_history ,training_history_his_type1 = 'std_auc' ,training_history_his_type2 ='val_std_auc'):
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(training_history.history[training_history_his_type1])
        plt.plot(training_history.history[training_history_his_type2])
        plt.title('Traing History(unit:epoch)')
        plt.ylabel('acc(unit:std)')    
        plt.legend(['train','validation'],loc = 'upper left')

        plt.subplot(212)
        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('Traing History(unit:epoch)')
        plt.ylabel('loss')
        plt.legend(['loss','val_loss'],loc = 'upper left')

        plt.savefig(name+'.fuzzy.history.png')
        plt.show()

