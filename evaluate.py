
import pandas as pd
import numpy as np


def getHit(df):
    if sum(df['pred']) < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    if sum(df['pred']) < _K:
        return 1 / np.log2(sum(df['pred']) + 2)
    else:
        return 0.


def getMRR(df):
    return 1 / (sum(df['pred']) + 1)


def evaluate_model(model, test, K):

    global _K
    _K = K
    test_X = test
    pos_score, neg_score = model.predict(test_X)
    test_df = pd.DataFrame(test_X[0], columns=['user_id'])
    if model.mode == 'inner':
        test_df['pred'] = (pos_score <= neg_score).astype(np.int32)
    else:
        test_df['pred'] = (pos_score >= neg_score).astype(np.int32)
    tg = test_df.groupby('user_id')
    hit_rate = tg.apply(getHit).mean()
    ndcg = tg.apply(getNDCG).mean()
    mrr = tg.apply(getMRR).mean()
    return hit_rate, ndcg, mrr