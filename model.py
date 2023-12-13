

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dropout
from tensorflow.keras.regularizers import l2,l1,l1_l2

from modules import *





# SelfAttention_Layer

class RPSANRec(Model):
    def __init__(self, feature_columns, maxlen=5, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6, **kwargs):

        super(RPSANRec, self).__init__(**kwargs)

        self.maxlen = maxlen

        self.w = w
        self.gamma = gamma
        self.mode = mode

        self.user_fea_col, self.item_fea_col = feature_columns

        self.embed_dim = self.item_fea_col['embed_dim']

        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

        # self.dropout = Dropout(0.5)
        self.self_attention = SelfAttention_Layer()
        self.bi_lstm_layer = Bi_Lstm_layer()

    def call(self, inputs, **kwargs):
        # input
        user_inputs, seq_inputs, pos_inputs, neg_inputs = inputs
        # mask
        # mask = self.item_embedding.compute_mask(seq_inputs)
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)

        # item
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim) 128, 100
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim) 128, 100
        # item2 embed
        pos_embed2 = self.item2_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)   128, 100
        neg_embed2 = self.item2_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)   128, 100

        # short-term interest
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)
        long_interest = self.bi_lstm_layer([seq_embed, seq_embed, seq_embed, mask])


        # mode
        if self.mode == 'inner':
            pos_long_interest = tf.multiply(long_interest, pos_embed2) # (128, 100)
            neg_long_interest = tf.multiply(long_interest, neg_embed2) # (128, 100)

            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            pass
        return pos_scores, neg_scores

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs], 
            outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    seq_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, seq_features]
    model = RPSANRec(features, mode='dist')
    model.summary()
