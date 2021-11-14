from __future__ import print_function

import os
import time
import datetime
import random
import json
import argparse
#import densenet
import csv
import cv2
import numpy as np
import keras.backend as K

from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils






import tensorflow as tf
print(tf.test.gpu_device_name())






def loadData(csvFilePath, size=512):
    Images = []
    Labels = []
    
    csvRows = []
    with open(csvFilePath, 'rt') as f:
        csvData = csv.reader(f)
        for row in csvData :
            csvRows.append((row[0], int(row[1])))
            
    random.shuffle(csvRows)
    
    for csvRow in csvRows:
        path = csvRow[0]
        files = [path + f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for filepath in files:
            try:
                image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image,(size,size))
                #image = randome_rotation_flip(image,size)
                Images.append(image)
                Labels.append(csvRow[1])

            except Exception as e:
                print(str(e))
                
        #if len(Labels) > 30:
            #break

    #Images = np.asarray(Images).astype('float32')

    mean = np.mean(Images)            #normalization
    std = np.std(Images)
    Images = (Images - mean) / std

    if K.image_data_format() == "channels_first":
        Images = np.expand_dims(Images,axis=1)           #Extended dimension 1
    if K.image_data_format() == "channels_last":
        Images = np.expand_dims(Images,axis=3)             #Extended dimension 3(usebackend tensorflow:aixs=3; theano:axixs=1) 
    return Images, Labels







from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def conv_factory(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :parameter x: Input keras network
    :parameter concat_axis: int -- index of contatenate axis
    :parameter nb_filter: int -- number of filters
    :parameter dropout_rate: int -- dropout rate
    :parameter weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :return type: keras network
    """
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :parameter x: keras model
    :parameter concat_axis: int -- index of contatenate axis
    :parameter nb_filter: int -- number of filters
    :parameter dropout_rate: int -- dropout rate
    :parameter weight_decay: int -- weight decay factor
    :returns: model
    :return type: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """
    Build a denseblock where the output of each conv_factory is fed to subsequent ones
    :parameter x: keras model
    :parameter concat_axis: int -- index of contatenate axis
    :parameter nb_layers: int -- the number of layers of conv_factory to append to the model.
    :parameter nb_filter: int -- number of filters
    :parameter dropout_rate: int -- dropout rate
    :parameter weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :return type: keras model
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate
        #print (nb_filter)

    return x, nb_filter


def denseblock_altern(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory is fed to subsequent ones. (Alternative of denseblock)
    :parameter x: keras model
    :parameter concat_axis: int -- index of contatenate axis
    :parameter nb_layers: int -- the number of layers of conv_factory to append to the model.
    :parameter nb_filter: int -- number of filters
    :parameter dropout_rate: int -- dropout rate
    :parameter weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :return type: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, concat_axis, growth_rate,  dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ 
    Build the DenseNet model
    :parameter nb_classes: int -- number of classes
    :parameter img_dim: tuple -- (channels, rows, columns)
    :parameter depth: int -- how many layers
    :parameter nb_dense_block: int -- number of dense blocks to add to end
    :parameter growth_rate: int -- number of filters to add
    :parameter nb_filter: int -- number of filters
    :parameter dropout_rate: float -- dropout rate
    :parameter weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :return type: keras model
    """
    
    if K.image_data_format() == "channels_first":
        concat_axis = 1
    elif K.image_data_format() == "channels_last":
        concat_axis = -1

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    nb_layers1 = [6,12,32,32,48,32,48,64,32]  #3*3 convolutional layer of each denseblock ，
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers1[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition
        x = transition(x, concat_axis,nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, concat_axis, nb_layers1[nb_dense_block-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes, activation='sigmoid', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return densenet






im_size = 480

print("Started Data Loading..")
X_train, Y_train = loadData('MURA-v1.1/train_labeled_studies.csv', size = im_size)
print("Training images: ", len(Y_train))
X_valid, Y_valid = loadData('MURA-v1.1/valid_labeled_studies.csv', size = im_size)
print("Validation images: ", len(Y_valid))

Y_valid = np.asarray(Y_valid)






def run_MURA(batch_size, nb_epoch, depth, nb_dense_block, nb_filter, growth_rate, dropout_rate, learning_rate, weight_decay, plot_architecture):
    """
    Run MURA experiments
    :parameter batch_size: int -- batch size
    :parameter nb_epoch: int -- number of training epochs
    :parameter depth: int -- network depth
    :parameter nb_dense_block: int -- number of dense blocks
    :parameter nb_filter: int -- initial number of conv filter
    :parameter growth_rate: int -- number of new filters added by conv layers
    :parameter dropout_rate: float -- dropout rate
    :parameter learning_rate: float -- learning rate
    :parameter weight_decay: float -- weight decay
    :parameter plot_architecture: bool -- whether to plot network architecture
    """

    ###################
    # Data processing #
    ###################

    
    #im_size = 320   #Test modification parameters size root_path nb_epoch nb_dense_block         
    #X_train_path, Y_train = data_loader.load_path(root_path = './train/XR_HUMERUS',size = im_size)  
    #X_valid_path, Y_valid = data_loader.load_path(root_path = './valid/XR_HUMERUS', size = im_size) 
    
    
    nb_classes = 1                                
    img_dim = (im_size,im_size,1)                 #Plus the last dimension, type is tuple

    
    ###################
    # Construct model #
    ###################

    model = DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)
    # Model output
    #model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    if plot_architecture:
        from keras.utils import plot_model
        plot_model(model, to_file='./try3/figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    print("Start Training")

    list_train_loss = []
    list_valid_loss = []
    list_learning_rate = []
    best_record = [100,0,100,100] #Recording optimal [verification set loss function value, accuracy rate, training set data set loss difference,acc difference]
    start_time = datetime.datetime.now()
    for e in range(nb_epoch):

        if e == int(0.25 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 50.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))


        split_size = batch_size
        num_splits = len(Y_train) / split_size

        arr_all = np.arange(len(X_train)).astype(int)
        #random.shuffle(arr_all)                 #Randomly disrupted index order
        arr_splits = np.array_split(arr_all, num_splits)

        l_train_loss = []
        batch_train_loss = []
        start = datetime.datetime.now()

        for i,batch_idx in enumerate(arr_splits):


            X_batch,Y_batch = [],[]
            for idx in batch_idx:
                X_batch.append(X_train[idx])
                Y_batch.append(Y_train[idx])
            
            X_batch = np.asarray(X_batch).astype('float32')
            Y_batch = np.asarray(Y_batch)
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])
            batch_train_loss.append([train_logloss, train_acc])
            if i %100 == 0:
                loss_1, acc_1 = np.mean(np.array(l_train_loss), 0)
                loss_2, acc_2 = np.mean(np.array(batch_train_loss), 0)
                batch_train_loss = []           #Current 100 batch loss function and accuracy
                print ('[Epoch {}/{}] [Batch {}/{}] [Time: {}] [all_batchs--> train_epoch_logloss: {:.5f}, train_epoch_acc:{:.5f}] '.format(e+1,nb_epoch,i, len(arr_splits),datetime.datetime.now() - start,loss_1,acc_1), '[this_100_batchs-->train_batchs_logloss: {:.5f}, train_batchs_acc:{:.5f}]'.format(loss_2, acc_2))

        # Run verification set
        valid_logloss, valid_acc = model.evaluate(X_valid, Y_valid, verbose=0, batch_size=64)
        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_valid_loss.append([valid_logloss, valid_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('[Epoch %s/%s] [Time: %s, Total_time: %s]' % (e + 1, nb_epoch, datetime.datetime.now() - start, datetime.datetime.now() - start_time),end = '')
        print('[train_loss_and_acc:{:.5f} {:.5f}] [valid_loss_acc:{:.5f} {:.5f}]'.format(list_train_loss[-1][0], list_train_loss[-1][1],list_valid_loss[-1][0],list_valid_loss[-1][1]))


        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        #d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["valid_loss"] = list_valid_loss
        d_log["learning_rate"] = list_learning_rate
        
        #print(d_log)

        json_file = os.path.join('./try3/log/experiment_log_MURA.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)



        record = [valid_logloss,valid_acc,abs(valid_logloss-list_train_loss[-1][0]),abs(valid_acc-list_train_loss[-1][1]),]
        if ((record[0]<=best_record[0]) &(record[1]>=best_record[1])) :
            if e <= int(0.25 * nb_epoch)|(record[2]<=best_record[2])&(record[3]<=best_record[3]):#Add a difference judgment after a quarter epoch
                best_record=record                      #Record the smallest [validation set loss function value, accuracy rate, training set data loss difference, acc difference]
                print('saving the best model:epoch',e+1,best_record)
                model.save('try3/save_models/best_MURA_modle@epochs{}.h5'.format(e+1))
        model.save('try3/save_models/MURA_try3_modle@epochs{}.h5'.format(e+1))
        model.save('try3/save_models/MURA_try3_modle@epochs{}_saved_model'.format(e+1))
        
    model.save('try3/saved_models/MURA_try3_modle@epochs_52_final_saved_model')
        
        
        
        
        
        


list_dir = ["./try3/log", "./try3/figures", "./try3/save_models"]
for d in list_dir:
    if not os.path.exists(d):
        os.makedirs(d)

run_MURA(8, 52, 6*3+4, 4,  16, 12, 0.2, 1E-3, 1E-4, False)







print("Done.....!!!!!!!")

