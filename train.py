import os
import numpy as np
import pandas as pd
import tensorflow as tf
from logging import getLogger
import warnings
import  matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

from time import time
from tensorflow.keras.optimizers import Adam

from model import RPSANRec
from modules import *
from evaluate import *
from util import *



if __name__ == '__main__':
    file = '../dataset/ml-100k/u.data'
    trans_score = 1
    maxlen = 5
    embed_dim = 100
    embed_reg = 1e-6
    gamma = 0.5
    mode = 'inner'
    w = 0.4
    K = 10
    learning_rate = 0.001
    epochs = 100
    batch_size = 256
    ndcg_list = []
    hit_list = []
    mrr_list = []
    val_loss = []
    loss = []
    feature_columns, train, val, test = datasets_preprocessing(file, trans_score, embed_dim, maxlen)
    train_X = train
    val_X = val
    model = RPSANRec(feature_columns, maxlen, mode, gamma, w, embed_reg)
    logger = getLogger()
    logger.info(model)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):

        t1 = time()
        tf.config.run_functions_eagerly(True)
        history = model.fit(
            train_X,
            None,
            validation_data=(val_X, None),
            epochs=1,
            # callbacks=[tensorboard, checkpoint],
            batch_size=batch_size,
            )
        t2 = time()

        hit_rate, ndcg, mrr = evaluate_model(model, test, K)

        hit_list.append(hit_rate)
        ndcg_list.append(ndcg)
        mrr_list.append(mrr)
        val_loss.append(history.history['val_loss'])
        loss.append(history.history['loss'])
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: Hit = %.4f, NDCG = %.4f, MRR = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr])

