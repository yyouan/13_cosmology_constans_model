import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense,Input,concatenate
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks
import numpy as np

'''

    main function : do all trainning here

'''
def model_train( name,input_array,output_array,training,data,with_graph):
    
    print('================== In model trainning =====================')
    '''

    layer construction : we need to tune them
    non-serial model : revised RBM degsign

    '''
    inputLen = input_array.shape[1]
    ## provide a normal distribution
    ## we need tune stddev for a good scaling
    normal = initializers.RandomNormal(mean=0.0, stddev=0.2, seed=10)

    input_layer = Input(shape=(inputLen,))
    dense_1 = Dense(units = 1000 , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu' ,name='forward_1')(input_layer)    
    dense_2 = Dense(units = 100 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu' ,name='forward_2')(dense_1)

    output_layer = Dense( units = output_array.shape[1] , 
                    kernel_initializer = normal , activation = 'linear',name='output') (dense_2) #only positive value
    degenerate_layer = Dense( units = int(np.abs( inputLen - output_array.shape[1] )) , 
                    kernel_initializer = normal , activation = 'linear',name='degenerate') (dense_2)
    med_layer = concatenate([output_layer,degenerate_layer])

    dense_3 = Dense(units = 100 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu', name='backward_1')(med_layer)
    dense_4 = Dense(units = 1000 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu', name='backward_2')(dense_3)
    recover_layer = Dense( units = inputLen , 
                    kernel_initializer = normal , activation = 'linear',name='recover') (dense_4)

    model = Model(inputs=input_layer,outputs=[recover_layer,output_layer])
    
    '''

    trainning

    '''
    
    #main code:

    #showing the prediction of last 5 data for scaling adjusting
    print('==========showing the prediction of last 5 data==========')
    training['prediction'] =  model.predict(input_array[-5:-1,:]) 
    print(training['prediction'])
    print('=========================================================')
    
    ##compile model
    ## lr is learing rate : we need to tune it for good learning
    ## loss is loss function ,metric is only for show how good is the trainging result
    ## mse: mean squred error , mae :mean absolute error
    adam = optimizers.Adam(lr=0.0003, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    model.compile(optimizer=adam,
                loss= 'mse',
                metrics={'recover':'mae','output':'mae'})
    
    ##time evalution:
    import time
    start_time = time.time()

    ##training mode
    # tensorboard viewing browser
    # checkpoint : save weight every period
    training['Callbacks'] = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    filepath ="log/"+ name + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    training['Checkpoint'] = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    training['History'] = model.fit(input_array,[ np.zeros((input_array.shape[0],inputLen)) ,output_array]
                                ,validation_split = training['Validation_split']
                                ,epochs = training['Epochs']
                                ,batch_size=training['BatchSize'] ,verbose=1 , callbacks = []) #verbose for show training process

    print("trainning cost :: --- %s seconds ---" % (time.time() - start_time))

    # serialize model to JSON
    model_json = model.to_json()
    with open("log/"+ name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights( "log/"+ name+".h5")
    print("Saved model to disk")

    if with_graph == True:
        show_train_history(name,training['History'])
 
    return model

def show_train_history(name,training_history):
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(training_history.history['recover_mean_absolute_error'])
        plt.plot(training_history.history['val_recover_mean_absolute_error'])
        plt.title('recover Traing History(unit:epoch)')
        plt.ylabel('acc(unit:std)')    
        plt.legend(['train','validation'],loc = 'upper left')

        plt.subplot(312)
        plt.plot(training_history.history['output_mean_absolute_error'])
        plt.plot(training_history.history['val_output_mean_absolute_error'])
        plt.title('output Traing History(unit:epoch)')
        plt.ylabel('acc(unit:std)')    
        plt.legend(['train','validation'],loc = 'upper left')

        plt.subplot(313)
        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('Traing History(unit:epoch)')
        plt.ylabel('loss')
        plt.legend(['loss','val_loss'],loc = 'upper left')

        plt.savefig(name+'.degenerate_vec.history.png')
        plt.show()
        

