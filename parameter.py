import numpy as np

'''

for data_mask.py and command --mask:

'''
# number of bin per dimension
mask_index = [(1-1),(2-1)]

'''

for data_thinned.py and command --thinned:

'''
# number of bin per dimension
crteria_index = (7 - 1)
criteria_medium = 91.18

'''

for data_transform.py and command --transform:

    type introction:

    log                : x -> log(x)
    normalise          : (x - mean)/std
    absmax_to_five     : max(abs(x)) -> 5
                         x -> x/max(abs(x))
    log_negative_2_dim : x -> log(abx(x))
                         new_dim (of the index) = tanh(x)
    negative_exp       : x -> exp(-x)
    exp                : x -> exp(x)

'''
## transform type of every data index

operation_array = {
    0:['log','normalise'], 
    1:['log','normalise'],
    2:['log','normalise'],
    3:['log','normalise'],
    4:['none'],
    5:['none'],
    6:['none'],
    7:['none'],
    8:['none'],
    9:['none'],
    10:['none'],
    11:['none'],
    12:['none'],
    13:['none'],
    14:['none'],
    15:['none'],
    16:['none'],
    17:['none'],
    18:['none'],
    19:['none'],
    20:['none'],
    21:['none'],
    22:['none'],
    23:['none'],
    24:['none'],
    25:['none'],
    26:['none'],
    27:['none'],
    28:['none'],
    29:['none'],
    30:['none'],
    31:['none'],
    32:['none'],
    33:['none'],
    34:['none'],
    35:['none'],
    36:['none'],
    37:['none'],
    38:['none'],
    39:['none'],
    40:['none'],
    
}

'''

for backward_training or command -bt / --backward_train
1.rate and region mean training part of the data
eg. Rate=0.1 Region=0.2 mean data[(Len(data)*0.2):(Len(data)*0.3),:] 
2.Epochs mean the trainning epoch, more epoch ,cost more time ,and learning better
3.Valiatdtion rate = 0.2 mean use 1/5 data to check the result of the learning made by other 4/5 data

'''
backward_training = {}
backward_training['Rate'] = 1.0
backward_training['Region'] = 0.0
backward_training['Validation_split'] = 0.2
backward_training['Epochs'] = 100
backward_training['BatchRate'] = 0.1
backward_inputLen = 4

## if you want to revise model , check the file src/backward_train_model.py

'''

for forward_training or command -ft / --forward_train
1.rate and region mean training part of the data
eg. Rate=0.1 Region=0.2 mean data[(Len(data)*0.2):(Len(data)*0.3),:] 
2.Epochs mean the trainning epoch, more epoch ,cost more time ,and learning better
3.Valiatdtion rate = 0.2 mean use 1/5 data to check the result of the learning made by other 4/5 data

'''
forward_training = {}
forward_training['Rate'] = 1.0
forward_training['Region'] = 0.0
forward_training['Validation_split'] = 0.2
forward_training['Epochs'] = 10
forward_training['BatchRate'] = 0.1
forward_inputLen = 4

## if you want to revise model , check the file src/forward_train_model.py

'''

for fuzzy_training or command -fzt / --fuzzy_train
1.rate and region mean training part of the data
eg. Rate=0.1 Region=0.2 mean data[(Len(data)*0.2):(Len(data)*0.3),:] 
2.Epochs mean the trainning epoch, more epoch ,cost more time ,and learning better
3.Valiatdtion rate = 0.2 mean use 1/5 data to check the result of the learning made by other 4/5 data

'''
fuzzy_training = {}
fuzzy_training['Rate'] = 1.0
fuzzy_training['Region'] = 0.0
fuzzy_training['Validation_split'] = 0.2
fuzzy_training['Epochs'] = 10
fuzzy_training['BatchRate'] = 0.1
fuzzy_training['Input_Partion'] = 8
fuzzy_training['Output_Partion'] = 4
fuzzy_inputLen = 11

## if you want to revise model , check the file src/fuzzy_train_model.py
## if you want to fuzzy method , check the file src/fuzzy_fuzzy_function.py

'''

for degenerate_vec_training or command -dvt / --den_vec_train
1.rate and region mean training part of the data
eg. Rate=0.1 Region=0.2 mean data[(Len(data)*0.2):(Len(data)*0.3),:] 
2.Epochs mean the trainning epoch, more epoch ,cost more time ,and learning better
3.Valiatdtion rate = 0.2 mean use 1/5 data to check the result of the learning made by other 4/5 data

'''
degenerate_vec_training = {}
degenerate_vec_training['Rate'] = 1.0
degenerate_vec_training['Region'] = 0.0
degenerate_vec_training['Validation_split'] = 0.2
degenerate_vec_training['Epochs'] = 10
degenerate_vec_training['BatchRate'] = 0.1
degenerate_vec_inputLen = 11
