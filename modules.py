import tensorflow as tf
import torch.nn as nn
import numpy as np
import math

from tensorflow.keras.layers import Layer, Dense,Dropout,BatchNormalization
from tensorflow.keras.losses import Loss


class SelfAttention_Layer(Layer):
    def __init__(self):
        super(SelfAttention_Layer, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(shape=[self.dim, self.dim], name='weight', 
            initializer='random_uniform')
        self.dropout = Dropout(0.)
        self.bn = BatchNormalization()
    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs

        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        q = tf.nn.relu(tf.matmul(q, self.W))
        k = tf.nn.relu(tf.matmul(k, self.W))
        mat_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.dim, dtype=tf.float32)
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, q.shape[1], 1])
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)
        outputs = self.bn(outputs)
        outputs = tf.nn.softmax(logits=outputs, axis=-1)
        outputs = tf.matmul(outputs, v)
        outputs = tf.reduce_mean(outputs, axis=1)
        return outputs

        Tensor("att_rec/embedding_1/embedding_lookup/Identity_1:0", shape=(None, 5, 100), dtype=float32)

    def positional_encoding(self, QK_input):
        inputs = tf.expand_dims(QK_input[0],0)
        outputs = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(100, return_sequences=True),merge_mode='mul'
        )(inputs)
        return outputs


    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

class Bi_Lstm_layer(Layer):
    def __init__(self):
        super(Bi_Lstm_layer, self).__init__()

    def call(self, inputs, **kwargs):
        front,last,a,b = inputs
        front = tf.tanh(front)
        last = tf.tanh(last)
        front= tf.nn.softmax(front)
        last = tf.nn.softmax(last)
        front += self.Bilstm(front)
        last += self.Bilstm(front)
        outputs = tf.reduce_mean(front+last, axis=1)
        return outputs

    def Bilstm(self, input_X):
        inputs = tf.expand_dims(input_X[0],0)
        outputs = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, return_sequences=True),merge_mode='mul'
        )(inputs)
        return outputs

