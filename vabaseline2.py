import tensorflow as tf
from scipy.io import wavfile
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import signal
import sys, os
import random
from create_dataset import create_dataset, create_dataset_addSynth
from random import shuffle
from eval_fm4 import evaluate_batch

def m_activation(x):
  #  return tf.sign(x)*tf.minimum(tf.abs(x),4)
  return tf.tanh(x)*4

def conv1d(x,f,k,s,act,tr):
    out = tf.layers.conv1d(x,filters=f,
             kernel_size=k,strides=s,activation=act,padding='same',
             kernel_initializer=tf.contrib.layers.xavier_initializer(),trainable=tr)
    return tf.layers.batch_normalization(out)

def conv2d_t(x,f,k,s,act,tr):
    out = tf.layers.conv2d_transpose(x,filters=f,padding="SAME",
             kernel_size=(k,1),strides=[s,1],activation=act,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),trainable=tr)
    return tf.layers.batch_normalization(out)


def generate(inputx,sample_length,init_conv=None,init_deconv=None,n_hidden=3,trable=True):
        inputx = tf.reshape(inputx,(-1,sample_length,1))
        activation = m_activation
        current_layer = inputx

        for f in [256,256,256,512]:
            current_layer = conv1d(current_layer,f,9,2,tf.tanh,tr=trable)
        current_layer = conv1d(current_layer,1024,5,2,tf.tanh,tr=trable)
        current_layer = conv1d(current_layer,1,1,1,None,tr=trable)

        current_layer = tf.contrib.layers.flatten(current_layer)

        mn = tf.layers.dense(current_layer, n_hidden, activation = None, trainable=trable)
        sd = 0.5 * tf.layers.dense(current_layer, n_hidden, activation = None)
        epsilon = tf.random_normal([tf.shape(current_layer)[0], n_hidden], 0.0, 4.0)

        z  = mn + tf.multiply(epsilon, tf.exp(sd))

        current_layer = z
        current_layer = tf.layers.dense(current_layer, sample_length/32/2, trainable=trable)
        current_layer = tf.layers.dense(current_layer, sample_length/32, trainable=trable)
        current_layer=tf.reshape(current_layer,(-1,sample_length/32,1,1))

        current_layer = conv2d_t(current_layer, 1024,5,2,tf.tanh,tr=trable)
        for f in [512,256,256,256]:
             current_layer = conv2d_t(current_layer,f,9,2,tf.tanh,tr=trable)
        current_layer = conv2d_t(current_layer,1,1,1,None,tr=trable)
        current_layer = tf.contrib.layers.flatten(current_layer)
        current_layer = tf.layers.dense(current_layer, sample_length, trainable=trable)

        output_layer=tf.reshape(current_layer,(-1,sample_length,1))
        return output_layer, z ,inputx, mn , sd
