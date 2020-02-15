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
        text_id = tokenizer.encode(text, add_special_tokens=False, max_length=128)
        text_id = [tokenizer.cls_token_id] + text_id[:126] + [tokenizer.sep_token_id]
        mask_attention = [1] * len(text_id)

        text_id = torch.LongTensor(text_id).unsqueeze(0)
        mask = torch.LongTensor(mask_attention).unsqueeze(0)

        output = model.bert(text_id, attention_mask=mask)
        v1 = output[0][0, 0].detach().cpu().numpy()

        return v1



def search_vectors(query, doc_vecs, docs):
    ## 単位ベクトル
    doc_vecs_len = np.linalg.norm(doc_vecs, axis=1)
    doc_vecs = doc_vecs / np.expand_dims(doc_vecs_len, 1)

    query /= np.linalg.norm(query)

    ##ロードした文書ベクトルとクエリーとでコサイン類似度を計算
    result = np.dot(doc_vecs, query.T)
    top_5 = result.argsort()[::-1][:5]

    for n in top_5:
        print(re.sub("\n", "", docs[n]))
        print("-" * 50)



# docs = pickle.load(open("./docs.pkl", "rb"))
# labels = pickle.load(open("./labels.pkl", "rb"))
#
#
# MODELS = [(BertForSequenceClassification, BertJapaneseTokenizer, "bert-base-japanese")]
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     # config = AlbertConfig.from_pretrained(pretrained_weights)
#     config = BertConfig.from_pretrained(pretrained_weights)
#
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights, config=config)
#
# import numpy as np
#
# doc_vecs = np.load("./doc_vecs.npy")
# doc_vecs_len = np.linalg.norm(doc_vecs, axis=1)
# doc_vecs = doc_vecs / np.expand_dims(doc_vecs_len, 1)
#
# model.eval()
#
# with torch.no_grad():
#     text = str(input("にゅうりょくしてください: "))
#     text_id = tokenizer.encode(text, add_special_tokens=False, max_length=128)
#     text_id = [tokenizer.cls_token_id] + text_id[:126] + [tokenizer.sep_token_id]
#     mask_attention = [1] * len(text_id)
#
#     text_id = torch.LongTensor(text_id).unsqueeze(0)
#     mask = torch.LongTensor(mask_attention).unsqueeze(0)
#
#     output = model.bert(text_id, attention_mask=mask)
#
# v1 = output[0][0, 0].detach().cpu().numpy()
# v1 /= np.linalg.norm(v1)
#
# result = np.dot(doc_vecs, v1.T)
# top_5 = result.argsort()[::-1][:5]
# print(top_5)
#
# import re
#
# for n in top_5:
#     print(re.sub("\n", "", docs[n]))
#     print(labels[n])
#     print("-" * 50)
