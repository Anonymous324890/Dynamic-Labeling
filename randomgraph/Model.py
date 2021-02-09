#-*-coding:utf-8-*-
import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib import *

def reduce_rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))

class Model:
    def mlp (self, input, hidden_size, activate):
      input = tf.layers.dense(input, hidden_size)
      input = activate(input)
      #input = tf.layers.dense(input, self.embedding_size)
      return input

    def count2(self):
        print ("parameters")
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def weights_nonzero(self, labels):
    #"""Assign weight 1.0 to all labels except for padding (id=0)."""
        return tf.to_float(tf.not_equal(labels, 0))

    def __init__(self, classnum, embedding_size,
                 batch_size, len_var):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.len_var = len_var
        
        self.global_step=tf.Variable(1, trainable=False, name="global_step") 
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.input_D = tf.placeholder(tf.float32, shape=[None, len_var, len_var])
        self.input_list = tf.placeholder(tf.int32, shape=[None, len_var])
        self.input_mask = tf.placeholder(tf.float32, shape=[None, len_var])
        self.input_mat = tf.placeholder(tf.float32, shape=[None, len_var, len_var])
        self.input_y = tf.placeholder(tf.int32, shape=[None])
        self.y_result = tf.one_hot(self.input_y, classnum)
        #self.y_result = tf.reduce_max(self.y_result, reduction_indices=[1])
        self.y_result = tf.cast(self.y_result, dtype=tf.float32)
        D_nonzeros = self.weights_nonzero(self.input_D) 
        D = self.input_D
        D_zeros = 1 - D_nonzeros#self.weights_nonzero(self.input_D) 
        D_1 = 1 / tf.math.sqrt(tf.reduce_sum(self.input_mat, reduction_indices = [1], keepdims=True) + D_zeros) * D_nonzeros
        D_2 = 1 / tf.math.sqrt(tf.reduce_sum(self.input_mat, reduction_indices = [2], keepdims=True) + D_zeros) * D_nonzeros
        #D_zeros = 1 - D_nonzeros#self.weights_nonzero(self.input_D) 
        #D = 1 / (tf.math.sqrt(self.input_D) + D_zeros) * D_nonzeros 
        mat = self.input_mat * D_1 * D_2 * D_nonzeros#tf.matmul(D, tf.matmul(self.input_mat, D))
        label_smoothing = 0
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / classnum
        self.y_result = self.y_result * smooth_positives + smooth_negatives
        
        self.embedding = tf.get_variable("embedding", [60 , embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
     
        em_list = tf.nn.embedding_lookup(self.embedding, self.input_list)
        em_list = em_list +  tf.expand_dims(-1e18 * (1 - self.input_mask), -1)
        
        for i in range(5):
          ori_list = em_list
          em_list = tf.transpose(tf.matmul(em_list, mat, transpose_a=True), perm=[0, 2, 1])
          em_list = self.mlp(em_list, self.embedding_size, tf.nn.relu)
        
        em_list = tf.reduce_max(em_list + tf.expand_dims(-1e18 * (1 - self.input_mask), -1), reduction_indices=[1])#em_list = tf.reduce_sum(em_list * tf.expand_dims(self.input_mask, -1), reduction_indices=[1]) / tf.reduce_sum(self.input_mask, reduction_indices=[1], keepdims=True)
        #em_list = tf.reduce_mean(em_list, reduction_indices=[1])

        # em_list = tf.mlp(em_list, self.embedding_size * 4, tf.nn.relu)
        em_list = tf.layers.dense(em_list, self.embedding_size * 4, activation=tf.nn.relu)
        em_list = tf.nn.dropout(em_list, self.keep_prob)
        All_q_a = tf.layers.dense(em_list, classnum)
        self.y = tf.nn.softmax(All_q_a)

        self.max_res = tf.argmax(self.y, 1)
        print ("max-res", self.max_res)
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_result, 1), tf.argmax(self.y, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        print (tf.shape(self.y_result))
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_result * tf.log(tf.clip_by_value(self.y, 1e-10, 1.0)), reduction_indices=[1])
            )
        
        self.loss = self.cross_entropy
        global_step = tf.cast(self.global_step, dtype=tf.float32)
        self.optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss , global_step=self.global_step)
