"""
helpers for tensorflow
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from Research_Platform.helpers import boiler_plate


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def reset_random_elements_keras():
    tf.keras.backend.clear_session()
    np.random.seed(123)
    tf.reset_default_graph()
    tf.set_random_seed(123)
    random.seed(123)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def tf_covariance(x, y):
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    N = tf.to_float(tf.shape(x)[0])
    return tf.reduce_sum(tf.multiply(x - mean_x, y - mean_y)) / N


def tf_boiler_plate(gpu):
    np.random.seed(123)
    tf.reset_default_graph()
    tf.set_random_seed(123)
    random.seed(123)
    tf.logging.set_verbosity('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    boiler_plate(gpu)


def reset_tf_session():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    return sess

def number_trainable_weights():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
