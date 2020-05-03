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
from tensorflow.keras.models import load_model


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


def model_transfer(input_shape, classes):
    X_input = Input(input_shape)

    X = Dense(256, input_dim=X_input.shape, activation='relu', name='fc_256',
              kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Dense(classes, input_dim=X.shape, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=X, name="model_transfer")
    return model


def Transfer(overlap_ratio_List_for_Target,
             overlap_ratio_List_for_Source,
             epochs_list,
             batch_size_list,
             range_of_class,
             Cube_size,
             Data,
             Gt,
             Train_Test_split,
             channel,
             Source_data_name,
             Target_data_name,
             Verbosity,
             LR = 0.0001):
    Accu_Table = [[None for l in range(len(overlap_ratio_List_for_Target) + 1)] for n in
                  range(len(overlap_ratio_List_for_Source) + 1)]
    for k in range(1, len(overlap_ratio_List_for_Target) + 1):
        Accu_Table[0][k] = Target_data_name + '_' + str(overlap_ratio_List_for_Target[k - 1])
    for k in range(1, len(overlap_ratio_List_for_Source) + 1):
        Accu_Table[k][0] = Source_data_name + '_' + str(overlap_ratio_List_for_Source[k - 1])
    Accu_Table[0][0] = "Test Accuracy"
    df_ini = pd.DataFrame.from_records(Accu_Table)
    print(df_ini)
    for m in range(len(overlap_ratio_List_for_Source)):
        for i in range(len(overlap_ratio_List_for_Target)):
            Source_Model = load_model('..\\..\\Trained Models\\Sub_Model\\'+Source_data_name+'\\Sub_model_Transfer_' +
                                      Source_data_name + '_overlap_ratio_' + str(overlap_ratio_List_for_Source[m]) +
                                      '_percent.h5')
            print(
                "\n\n===============================================================================================================================\n"
                "= Source Model with overlap ratio of " + str(overlap_ratio_List_for_Source[m]) +
                "% will be transfered to Target Data with overlap ratio of " + str(overlap_ratio_List_for_Target[i]) +
                "%, Source Model shown below =\n"
                "===============================================================================================================================\n\n")
            Source_Model.summary()
            overlap_ratio = overlap_ratio_List_for_Target[i] / 100
            Xtrain, Xtest, Ytrain, Ytest, class_len, counts = prepare_data_for_training(range_of_class=range_of_class,
                                                                                        Cube_size=Cube_size, Data=Data,
                                                                                        Gt=Gt, small_segmented_1=[],
                                                                                        small_seg_gt_1=[],
                                                                                        percentage=Train_Test_split,
                                                                                        overlap_ratio=overlap_ratio,
                                                                                        ch=channel)

            Xtrain_transfer = Source_Model.predict(Xtrain)
            Xtest_transfer = Source_Model.predict(Xtest)
            print('Xtrain_transfer => ' + str(Xtrain_transfer.shape) + '\n' +
                  'Xtest_transfer  => ' + str(Xtrain_transfer.shape) + '\n' +
                  'Ytrain => ' + str(Ytrain.shape) + '\n' +
                  'Ytest  => ' + str(Ytest.shape) + '\n')
            model_1 = model_transfer(input_shape=Xtrain_transfer[0].shape, classes=len(Ytrain[0]))
            model_1.summary()

            model_checkpoint = ModelCheckpoint('..\\..\\Trained Models\\Transfered_Models\\Fully_Connected_Layers_Only\\'
                + Target_data_name + '\\Fully_Connected_Layers_for_' + Target_data_name +
                '_overlap_ratio_' + str(int(overlap_ratio * 100)) +
                '_With_source_data_as_' + Source_data_name + '_overlap_ratio_' +
                str(int(overlap_ratio_List_for_Source[m])) + '_percent.h5',
                monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
            model_1.compile(optimizer=keras.optimizers.SGD(lr=LR, decay=1e-5, momentum=0.9, nesterov=True),
                            loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            model_1.fit(Xtrain_transfer, Ytrain, epochs=epochs_list[i], batch_size=batch_size_list[i],
                        validation_data=[Xtest_transfer, Ytest], verbose=Verbosity, callbacks=[model_checkpoint])

            preds = model_1.evaluate(Xtest_transfer, Ytest)
            # print ("Loss = " + str(preds[0]))
            print("Test Accuracy = " + str(preds[1]))
            Accu_Table[m + 1][i + 1] = preds[1] * 100
            y_pred = model_1.predict(Xtest_transfer, verbose=1)
            confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(Ytest, axis=1), np.argmax(y_pred, axis=1))
            print("\n===============================================================================================================================\n"
                "=========== Confusion_matrix for data with source overlap ratio of " +
                str(overlap_ratio_List_for_Source[m]) + "% and target overlap ratio of  " +
                str(overlap_ratio_List_for_Target[i]) + "% is as below ===========\n"
                                                        "===============================================================================================================================\n")
            print(confusion_matrix)
            print(
                "==============================================================================================================================")
            print(counts)
    df = pd.DataFrame.from_records(Accu_Table)

    return df