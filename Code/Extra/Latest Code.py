import scipy.io as sio
import numpy as np
import math
import keras
import sklearn.metrics
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling3D
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

from matplotlib.pyplot import imshow


def pick_class(Class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1):
    indx_class = np.where(Gt == Class)
    all_indx = [[indx_class[0][i], indx_class[1][i]] for i in range(len(indx_class[0])) if
                len(Gt) - np.ceil(Cube_size / 2) > indx_class[0][i] > np.ceil(Cube_size / 2) and len(Gt[0]) - np.ceil(
                    Cube_size / 2) > indx_class[1][i] > np.ceil(Cube_size / 2)]
    lst = []
    small_segmented_1.append(np.array(Data[:, all_indx[0][0] - int(Cube_size / 2):all_indx[0][0] + int(Cube_size / 2),
                                      (all_indx[0][1] - int(Cube_size / 2)):all_indx[0][1] + int(Cube_size / 2)]))
    small_seg_gt_1.append(Class)
    lst.append([all_indx[0][0], all_indx[0][1]])
    for i in range(1, len(all_indx)):
        dist = []
        for k in range(len(lst)):
            d = math.sqrt((all_indx[i][0] - lst[k][0]) ** 2 + (all_indx[i][1] - lst[k][1]) ** 2)
            dist.append(d)
        if np.min(dist) > int(Cube_size / 2):
            small_segmented_1.append(np.array(
                Data[:, all_indx[i][0] - int(Cube_size / 2):all_indx[i][0] + int(Cube_size / 2),
                (all_indx[i][1] - int(Cube_size / 2)):all_indx[i][1] + int(Cube_size / 2)]))
            small_seg_gt_1.append(Class)
            lst.append([all_indx[i][0], all_indx[i][1]])
    return small_segmented_1, small_seg_gt_1, lst


def pick_n_class(range_of_class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1):
    class_len = []
    for i in range_of_class:
        small_segmented_1, small_seg_gt_1, lst = pick_class(i, Cube_size, Data, Gt, small_segmented_1,
                                                            small_seg_gt_1)
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
    Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape
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


def prepare_data_for_training(range_of_class, Cube_size, Data, Gt, small_segmented_1, small_seg_gt_1, percentage):
    small_segmented_1, small_seg_gt_1, class_len = pick_n_class(range_of_class, Cube_size, Data, Gt, small_segmented_1,
                                                                small_seg_gt_1)
    Xtrain, Xtest, Ytrain, Ytest, class_len, counts = train_test_split(percentage, class_len, small_segmented_1,
                                                                       small_seg_gt_1)
    return Xtrain, Xtest, Ytrain, Ytest, class_len, counts


def SR_Unit(X, filters):
    # Save the input value
    X_shortcut = X
    print(X.shape)
    X = Conv3D(filters, (1, 1, 1), padding="same")(X)
    print('--------')
    print(X.shape)
    X = BatchNormalization(axis=4)(X)
    X = Activation('relu')(X)
    print(X.shape)

    X = Conv3D(filters, kernel_size=(1, 3, 3), padding='same')(X)
    X = BatchNormalization(axis=4)(X)
    X = Activation('relu')(X)
    print(X.shape)

    X = Conv3D(filters, kernel_size=(3, 1, 1), padding="same")(X)
    X = BatchNormalization(axis=4)(X)
    X = Activation('relu')(X)
    print(X.shape)

    X = Conv3D(2 * filters, kernel_size=(1, 1, 1), padding="same")(X)
    X = BatchNormalization(axis=4)(X)
    print(X.shape)

    X_shortcut = Conv3D(filters=2 * filters, kernel_size=(3, 3, 3), padding='same')(X_shortcut)
    X_shortcut = BatchNormalization(axis=4)(X_shortcut)

    X = Add()([X, X_shortcut])
    print(X.shape)
    print('--------')
    X = Activation('relu')(X)

    return X


def feature_extraction(Sample):
    X = Sample
    X = Conv3D(32, (8, 3, 3), strides=(2, 2, 2), name='conv1')(X)
    print(X.shape)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((1, 2, 2), strides=None)(X)
    print(X.shape)
    # Stage 2
    F1 = 16
    X = SR_Unit(X, filters=F1)
    print(X.shape)
    F2 = 32
    X = SR_Unit(X, filters=F2)
    print(X.shape)
    #     F3 = 64
    #     X = SR_Unit(X, filters=F3)
    #     print(X.shape)
    #     F4 = 128
    #     X = SR_Unit(X, filters=F4)
    print(X.shape)

    return X


def model(input_shape=(103, 50, 50, 1), classes=9):
    X_input = Input(input_shape)

    print(X_input.shape)
    X = feature_extraction(X_input)

    X = GlobalMaxPooling3D()(X)
    print(X.shape)
    X = Dense(classes, input_dim=256, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="3D-SRNet")

    return model

uIndian_pines = sio.loadmat('C:\\Users\\de991521\\Desktop\\EE_297_PROJECT\\Indian Pines\\Indian_pines_corrected.mat')
gt_uIndian_pines = sio.loadmat('C:\\Users\\de991521\\Desktop\\EE_297_PROJECT\\Indian Pines\\Indian_pines_gt.mat')
data_IN = uIndian_pines['indian_pines_corrected']
gt_IN = gt_uIndian_pines['indian_pines_gt']
print(data_IN.shape, gt_IN.shape)
data_IN = np.moveaxis(data_IN, 2, 0)
print(data_IN.shape, gt_IN.shape)


values,counts = np.unique(gt_IN, return_counts=True)
print(values,counts)
range_of_class = list(values)
if 0 in range_of_class:
    range_of_class.pop(0)
print(range_of_class)

Xtrain, Xtest, Ytrain, Ytest, class_len, counts = prepare_data_for_training(range_of_class = range_of_class, Cube_size = 25, Data = data_IN , Gt = gt_IN, small_segmented_1 = [], small_seg_gt_1 = [], percentage = 80)

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)