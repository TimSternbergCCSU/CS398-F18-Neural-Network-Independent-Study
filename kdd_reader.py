"""Utilities for parsing text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow as tf
import numpy as np
import random as rand

from tensorflow.python.client import device_lib

import hashlib

rand.seed()

flags = tf.flags
FLAGS = flags.FLAGS
logging = tf.logging

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.readlines()

# May want to specify Whether to store multi-dimensional data in row-major (C-style) or
# column-major (Fortran-style) order in memory. Specified in order. Figure out later.
def _build_np_array(data):
    for i,slice in enumerate(data):
        data[i] = np.array(slice.split(','))
    return data

def _attempt_cast_double(val):
    try:
        return double(val)
    except (ValueError, TypeError):
        return False

def _normalize_data(data):
    new_data = []
    for o,slice in enumerate(data):
        if rand.randint(1,10000)/100 <= float(FLAGS.percent_of_data):
            for i in range(0,len(slice)):
                value = None
                try:
                    value = int(slice[i])
                except (ValueError, TypeError):
                    m = hashlib.md5()
                    h = slice[i].encode('utf-8')
                    m.update(h)
                    value = int(str(int(m.hexdigest(), 16))[0:12]) % 1000
                slice[i] = value
            # Deleting the last element (type of anomoly)
            new_data.append( np.delete(slice,len(slice)-1) )
    return np.asarray(new_data)

def kdd_raw_data(data_path=None):
    full_data_path = os.path.join(data_path, "kddcup_data_10_percent_corrected.txt")
    unformatted_data = _read_words(full_data_path)
    unnormalized_data = _build_np_array(unformatted_data)
    data = _normalize_data(unnormalized_data)
    #print(data[0])
    print("Using",FLAGS.percent_of_data + "%","of dataset")
    data_len = len(data)
    train_data,valid_data,test_data = np.split(data,[int(data_len/4),int(data_len/2)])
    #print(len(train_data),len(valid_data),len(test_data))
    return train_data, valid_data, test_data, data_len

def kdd_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "KDDProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int64)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size // 41

        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len * 41])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
          epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
