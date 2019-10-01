import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks


'''

    main function : do all trainning here

'''
def model_train( name,input_array,output_array,training,data,output_index_array,with_graph):
    
    print('================== In model trainning =====================')
    '''

    layer construction : we need to tune them

    '''

    model = Sequential()
    inputLen = input_array.shape[1]
    ## provide a normal distribution
    ## we need tune stddev for a good scaling
    normal = initializers.RandomNormal(mean=0.0, stddev=0.1, seed=10)
    ## provide a uniform distribution
    ## we need tune stddev for a good scaling
    uniform = initializers.RandomUniform(minval=-0.3, maxval=0.3, seed=10)


    output_layer = Dense( units = 100 , input_dim = input_array.shape[1] 
                         , kernel_initializer = normal, bias_initializer=normal , activation = 'relu') #only positive value
    #BN is used to accelerate Sigmoid
    #BN_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    dense_3 = Dense(units = 1000 , input_dim = inputLen , kernel_initializer= normal,
                    bias_initializer=normal , activation = 'relu') 
    #dense_2 = Dense(units = 100, kernel_initializer=normal,
    #                bias_initializer=normal , activation = 'relu')       
    dense_1 = Dense( units = output_array.shape[1] , kernel_initializer= normal,
                    activation = 'linear')
    layer_list = [output_layer,dense_3,dense_1 ]   

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
    adam = optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.02, amsgrad=False)
    model.compile(optimizer=adam,
                loss= 'mse',
                metrics=['mae'])

    ##time evalution:
    import time
    start_time = time.time()
    ##training mode
    
    ##Callbacks:
    # tensorboard viewing browser
    # checkpoint : save weight every period
    training['Callbacks'] = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    filepath = "log/"+ name + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    training['Checkpoint'] = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
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
        show_train_history(name,training['History'],'mean_absolute_error','val_mean_absolute_error')
     
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

        plt.savefig("graph/"+name+'.forward.histroy.png')
        plt.show()

