import glob
import codecs
import pickle
from transformers import *
import MeCab

import torch
import numpy as np
import re


def extract_features(input_sentence, model, tokenizer):
    model.eval()
    with torch.no_grad():
        text = input_sentence

        # special_tokensを抜き、先頭128個のtokenをベクトル化
        text_id = tokenizer.encode(text, add_special_tokens=False, max_length=128)

        # special_tokensを追加
        text_id = [tokenizer.cls_token_id] + text_id[:126] + [tokenizer.sep_token_id]

        # mask_attentionを作成
        mask_attention = [1] * len(text_id)

        # 二次元ベクトルにする
        text_id = torch.LongTensor(text_id).unsqueeze(0)
        mask = torch.LongTensor(mask_attention).unsqueeze(0)

        # 推論して、cls_toke_idを取得
        output = model.bert(text_id, attention_mask=mask)
        v1 = output[0][0, 0].detach().cpu().numpy()
        return v1


def search_vectors(query, doc_vecs, docs):
    ## 単位ベクトル
    '''
    :param query: (768,)
    :param doc_vecs: (文書数,768)
    :param docs: list [文書1,文書2]
    :return:
    '''

    # 各文書の長さを配列で取得
    doc_vecs_len = np.linalg.norm(doc_vecs, axis=1)
    ## doc_vecs_len -> (文書数,)

    # 行 列合わして単位ベクトル化
    doc_vecs = doc_vecs / np.expand_dims(doc_vecs_len, 1)
    query /= np.linalg.norm(query)

    ##ロードした文書ベクトルとクエリとでコサイン類似度を計算(単位ベクトルかは済んでいるので、内積をとるだけ)
    result = np.dot(doc_vecs, query.T)
    top_100 = result.argsort()[::-1][:100]
    return_dict = []

    for n in top_100:
        list_doc = docs[n].split('\n')
        return_dict.append({"title": list_doc[2], "url": list_doc[0]})
    return_dicts = {"pages": return_dict}
    return return_dicts
