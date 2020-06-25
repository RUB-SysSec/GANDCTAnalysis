import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu",
                      padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_data])
    x = layers.Activation("relu")(x)

    return x


def build_resnet(input_shape, classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)

    for i in range(10):
        x = _res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    res_net_model = keras.Model(inputs, outputs)
    return res_net_model


def build_simple_cnn(input_shape, classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(3, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 64

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 32

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_simple_nn(input_shape, classes, l2=0.01):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2))(x)
    # x = layers.Dense(128, activation='relu',
    #                  kernel_regularizer=keras.regularizers.l2(l2))(x)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_multinomial_regression(input_shape, classes, kernel_regularizer=None, dataset=None):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation,
                           kernel_regularizer=kernel_regularizer)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_svm(input_shape, classes, l_2, logits=False):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(classes, activation="linear",
                     kernel_regularizer=keras.regularizers.l2(l_2))(x)

    if logits:
        return keras.Model(inputs, x)

    if classes > 1:
        outputs = 2 * layers.Softmax()(2*x) - 1
    else:
        outputs = layers.Activation(tf.nn.tanh)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_multinomial_regression_l2(input_shape, classes, l_2=0.01):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l2(l_2))


def build_multinomial_regression_l1(input_shape, classes, l_1=0.1):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l1(l_1))


def build_multinomial_regression_l1_l2(input_shape, classes, l_1=0.01, l_2=0.01):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l1_l2(l_1, l_2))
