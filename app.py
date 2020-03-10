import json

import falcon

from extract_features_pytorch import extract_features, search_vectors

from transformers import *

import numpy as np
import pickle


class AppResource(object):
    def __init__(self):
        # 学習時に使った日本語tokenizer等の用意
        MODELS = [(BertForSequenceClassification, BertJapaneseTokenizer, "bert-base-japanese")]
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            config = BertConfig.from_pretrained(pretrained_weights)

            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights, config=config)

        self.doc_vecs = np.load("./doc_vecs.npy")
        self.docs = pickle.load(open("./docs.pkl", "rb"))
        print("connect")

    def on_post(self, req, res):
        query = req.media.get('q')
        data = extract_features(query[0], self.model, self.tokenizer)
        response_values = search_vectors(data, self.doc_vecs, self.docs)
        res.body = json.dumps(response_values)


app = falcon.API()
app.add_route("/", AppResource())

if __name__ == "__main__":
    from wsgiref import simple_server

    httpd = simple_server.make_server("127.0.0.1", 8080, app)
    httpd.serve_forever()
