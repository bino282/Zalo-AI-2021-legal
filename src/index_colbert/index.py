import pandas as pd
import pyterrier as pt
import json
import re
import os

ROOT_DATA = "zalo_data/"
ColBERT_model_path = "models/colbert/colbert.dnn"
index_path = "legal_corpus_index/colbert/"
index_name = "legal.idx"

def preprocess(text):
    try:
        text = re.sub("\t"," ",text)
        text = re.sub("\n"," ",text)
        text = re.sub("^Điều [0-9][0-9]*?. "," ",text).strip()
        text = re.sub("^[0-9][0-9]*?. "," ",text).strip()
    except:
        return ""
    return text

if not pt.started():
  pt.init()

texts = []
doc_no = []

with open(os.path.join(ROOT_DATA,"legal_corpus.json"),"r",encoding="utf-8") as fr:
    legal_corpus = json.load(fr)
count = 0
for item in legal_corpus:
    for article in item["articles"]:
        count += 1
        texts.append(preprocess(article["title"])+ "[SEP]" + preprocess(article["text"]))
        doc_no.append(item["law_id"]+ "_" + article["article_id"])

df = pd.DataFrame({"text":texts,"docno":doc_no})
def legal_generate():
    for idx,row in df.iterrows():
        docno, passage = row["docno"], row["text"]
        yield {'docno' : docno, 'text' : passage}
data_gen = legal_generate()
from pyterrier_colbert.indexing import ColBERTIndexer
indexer = ColBERTIndexer(ColBERT_model_path,index_path,index_name,chunksize=64)
indexer.index(data_gen)

