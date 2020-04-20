import numpy as np
import pandas as pd
import math
import sklearn.metrics

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers


import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def pick_class(Class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1, overlap_ratio, ch, min_samples = 4):
    indx_class = np.where(Gt == Class)
    all_indx = [[indx_class[0][i], indx_class[1][i]] for i in range(len(indx_class[0])) if
                len(Gt) - np.ceil(Cube_size / 2) > indx_class[0][i] > np.ceil(Cube_size / 2) and len(Gt[0]) - np.ceil(
                    Cube_size / 2) > indx_class[1][i] > np.ceil(Cube_size / 2)]
    lst = []
    lst.append([all_indx[0][0], all_indx[0][1]])
    for i in range(1, len(all_indx)):
        dist = []
        for k in range(len(lst)):
            d = math.sqrt((all_indx[i][0] - lst[k][0]) ** 2 + (all_indx[i][1] - lst[k][1]) ** 2)
            dist.append(d)
        if np.min(dist) > int(Cube_size * (1 - overlap_ratio)):
            lst.append([all_indx[i][0], all_indx[i][1]])
    new_lst = []
    if len(lst)>=min_samples:
        small_segmented_1.append(
            np.array(Data[:ch, all_indx[0][0] - int(Cube_size / 2):all_indx[0][0] + int(Cube_size / 2),
                     (all_indx[0][1] - int(Cube_size / 2)):all_indx[0][1] + int(Cube_size / 2)]))
        small_seg_gt_1.append(Class)
        for i in range(1, len(lst)):
            small_segmented_1.append(np.array(Data[:ch, lst[i][0] - int(Cube_size / 2):lst[i][0] + int(Cube_size / 2), (lst[i][1] - int(Cube_size / 2)):lst[i][1] + int(Cube_size / 2)]))
            small_seg_gt_1.append(Class)
        new_lst = lst
    return small_segmented_1, small_seg_gt_1, new_lst


def pick_n_class(range_of_class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1, overlap_ratio, ch):
    class_len = []
    for i in range_of_class:
        small_segmented_1, small_seg_gt_1, lst = pick_class(i, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1, overlap_ratio, ch)
        class_len.append(len(lst))
    small_segmented_1 = np.array(small_segmented_1)
    small_seg_gt_1 = np.array(small_seg_gt_1)
    return small_segmented_1, small_seg_gt_1, class_len


def train_test_split(percentage, class_len, Data, Gt):
    Xtrain = []
    Xtest = []
    Ytrain = []
    Ytest = []
    class_division = [0]
    c = 0
    for i in range(len(class_len)):
        class_division.append(int(class_len[i] * (percentage / 100)) + c)
        class_division.append(class_len[i] + c)
        c = class_len[i] + c
    for i in range(1, len(class_division)):
        if i % 2 != 0:
            for j in range(class_division[i - 1], class_division[i]):
                Xtrain.append(Data[j])
                Ytrain.append(Gt[j])
        else:
            for k in range(class_division[i - 1], class_division[i]):
                Xtest.append(Data[k])
                Ytest.append(Gt[k])
    Xtrain = np.array(Xtrain)
    Xtest = np.array(Xtest)
    Ytrain = np.array(Ytrain)
    Ytest = np.array(Ytest)
    s = np.arange(Xtrain.shape[0])
    np.random.shuffle(s)
    Xtrain = Xtrain[s]
    Ytrain = Ytrain[s]
    s = np.arange(Xtest.shape[0])
    np.random.shuffle(s)
    Xtest = Xtest[s]
    Ytest = Ytest[s]
    Xtrain = np.expand_dims(Xtrain, axis=4)
    Xtest = np.expand_dims(Xtest, axis=4)
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    values, counts = np.unique(Ytest, return_counts=True)
    print(
        "Total samples per class: " + str(class_len) + ", Total number of samples is " + str(np.sum(class_len)) + '.\n')
    print("unique classes in Ytest: " + str(values) + ", Total number of samples in Ytest is " + str(
        np.sum(counts)) + '.\n'
          + "number of samples per class in Ytest: " + str(counts) + '\n')

    onehot_encoder = OneHotEncoder(sparse=False)
    Ytrain = Ytrain.reshape(len(Ytrain), 1)
    Ytest = Ytest.reshape(len(Ytest), 1)
    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.fit_transform(Ytest)

    return Xtrain, Xtest, Ytrain, Ytest, class_len, counts


def prepare_data_for_training(range_of_class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1, percentage,
                              overlap_ratio, ch):
    small_segmented_1, small_seg_gt_1, class_len = pick_n_class(range_of_class, Cube_size, Data, Gt, small_segmented_1,
                                                                small_seg_gt_1, overlap_ratio, ch)
    Xtrain, Xtest, Ytrain, Ytest, class_len, counts = train_test_split(percentage, class_len, small_segmented_1,
                                                                       small_seg_gt_1)
    return Xtrain, Xtest, Ytrain, Ytest, class_len, counts


def SR_Unit(X, filters):
    # Save the input value
    X_shortcut = X
    l2_ = 0.01
    X = Conv3D(filters, (1, 1, 1), kernel_regularizer=regularizers.l2(l2_), padding="same",
               name='SR_Unit_' + str(filters) + '_conv_1')(X)
    X = BatchNormalization(axis=4, name='SR_Unit_' + str(filters) + '_bn_1')(X)
    X = Activation('relu', name='SR_Unit_' + str(filters) + '_activation_1')(X)

    X = Conv3D(filters, kernel_size=(1, 3, 3), kernel_regularizer=regularizers.l2(l2_), padding='same',
               name='SR_Unit_' + str(filters) + '_conv_2')(X)
    X = BatchNormalization(axis=4, name='SR_Unit_' + str(filters) + '_bn_2')(X)
    X = Activation('relu', name='SR_Unit_' + str(filters) + '_activation_2')(X)

    X = Conv3D(filters, kernel_size=(3, 1, 1), kernel_regularizer=regularizers.l2(l2_), padding="same",
               name='SR_Unit_' + str(filters) + '_conv_3')(X)
    X = BatchNormalization(axis=4, name='SR_Unit_' + str(filters) + '_bn_3')(X)
    X = Activation('relu', name='SR_Unit_' + str(filters) + '_activation_3')(X)

    X = Conv3D(2 * filters, kernel_size=(1, 1, 1), kernel_regularizer=regularizers.l2(l2_), padding="same",
               name='SR_Unit_' + str(filters) + '_conv_4')(X)
    X = BatchNormalization(axis=4, name='SR_Unit_' + str(filters) + '_bn_4')(X)

    X_shortcut = Conv3D(filters=2 * filters, kernel_size=(3, 3, 3), padding='same',
                        kernel_regularizer=regularizers.l2(l2_), name='SR_Unit_' + str(filters) + '_conv_5')(X_shortcut)
    X_shortcut = BatchNormalization(axis=4, name='SR_Unit_' + str(filters) + '_bn_5')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu', name='SR_Unit_' + str(filters) + '_activation_5')(X)

    return X


def feature_extraction(Sample):
    X = Sample

    X = Conv3D(32, (8, 3, 3), kernel_regularizer=regularizers.l2(0.01), strides=(2, 2, 2), name='conv_0')(X)
    X = BatchNormalization(axis=4, name='bn_conv_0')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((1, 2, 2), strides=None, name='MaxPooling_0')(X)

    # Stage 2
    F1 = 16
    X = SR_Unit(X, filters=F1)

    F2 = 32
    X = SR_Unit(X, filters=F2)

    # F3 = 64
    # X = SR_Unit(X, filters=F3)

    #     F4 = 128
    #     X = SR_Unit(X, filters=F4)

    return X


def model(input_shape, classes):
    X_input = Input(input_shape)

    X = feature_extraction(X_input)
    X = GlobalMaxPooling3D()(X)
    X = Dense(256, input_dim=X.shape, activation='relu', name='fc_256', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, input_dim=X.shape, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="3D-SRNet")

    return model


def Train(overlap_ratio_List,
          range_of_class,
          Cube_size,
          Data,
          Gt,
          Train_Test_split,
          channel,
          epochs_list,
          batch_size_list,
          data_name,
          Verbosity):

    Accu_Table = [[None for l in range(2)] for n in range(len(overlap_ratio_List)+1)]
    for k in range(1,len(overlap_ratio_List)+1):
        Accu_Table[k][0] = 'Data_Overlap_' + str(overlap_ratio_List[k-1])
    Accu_Table[0][0] = 'Test Accuracy Table'
    Accu_Table[0][1] = data_name
    df_ini = pd.DataFrame.from_records(Accu_Table)
    print(df_ini)
    for i in range(len(overlap_ratio_List)):
        print("\n\n===============================================================================================================================\n"
              "======================================== Data with overlap ratio of "+ str(overlap_ratio_List[i]) + "% will be trained =======================================\n"
              "===============================================================================================================================\n\n")

        overlap_ratio = overlap_ratio_List[i] / 100
        Xtrain, Xtest, Ytrain, Ytest, class_len, counts = prepare_data_for_training(range_of_class=range_of_class,
                                                                                    Cube_size=Cube_size, Data=Data,
                                                                                    Gt=Gt, small_segmented_1=[],
                                                                                    small_seg_gt_1=[], percentage=Train_Test_split,
                                                                                    overlap_ratio=overlap_ratio, ch=channel)
        print('Xtrain => ' + str(Xtrain.shape) + '\n' +
              'Xtest  => ' + str(Xtest.shape) + '\n' +
              'Ytrain => ' + str(Ytrain.shape) + '\n' +
              'Ytest  => ' + str(Ytest.shape) + '\n')
        model_1 = model(input_shape=Xtrain[0].shape, classes=len(Ytrain[0]))
        model_1.summary()

        model_checkpoint = ModelCheckpoint(
            '..\\..\\Trained Models\\Full_Model\\'+data_name+'\\Full_Model_best_'+ data_name +'_overlap_ratio_' + str(int(overlap_ratio * 100)) + '_percent.h5',
            monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
        model_1.compile(optimizer=keras.optimizers.SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True),
                        loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model_1.fit(Xtrain, Ytrain, epochs=epochs_list[i], batch_size=batch_size_list[i],
                    validation_data=[Xtest, Ytest], verbose=Verbosity , callbacks=[model_checkpoint])

        preds = model_1.evaluate(Xtest, Ytest)
        # print ("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))
        Accu_Table[i+1][1] = preds[1] * 100
        y_pred = model_1.predict(Xtest, verbose=1)
        confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(Ytest, axis=1), np.argmax(y_pred, axis=1))
        print("\n===============================================================================================================================\n"
              "=============================== Confusion_matrix for data with overlap ratio of "+ str(overlap_ratio_List[i]) + "% is as below ===============================\n"
              "===============================================================================================================================\n")
        print(confusion_matrix)
        print("==============================================================================================================================")
        print(counts)
        print("==============================================================================================================================\n\n"
              "==============================================================================================================================\n"
              "=============================== Sub Model below will be saved to be used for transfer learning  ==============================\n"
              "==============================================================================================================================\n")
        model_1._layers.pop()
        model_1._layers.pop()
        # last_layer = model_1._layers.pop()
        # second_last_layer = model_1._layers.pop()
        model_2 =  Model(model_1.inputs, model_1.layers[-1].output)
        model_2.compile(optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True),
                        loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model_2.set_weights(model_1.get_weights())
        model_2.summary()

        model_2.save('..\\..\\Trained Models\\Sub_Model\\'+data_name+'\\Sub_model_Transfer_'+ data_name +'_overlap_ratio_' + str(int(overlap_ratio * 100)) + '_percent.h5')

    df = pd.DataFrame.from_records(Accu_Table)
    return df