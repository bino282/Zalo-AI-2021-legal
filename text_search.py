from rank_bm25 import *
import os
import re
import json
from tqdm import tqdm
import pandas as pd
from pyterrier_colbert.ranking import ColBERTFactory
os.environ['TOKENIZERS_PARALLELISM']='TRUE'
os.environ["JAVA_HOME"]="/usr/bin/java"
pytcolbert = ColBERTFactory(init_model="pretrained_model/electra-legal-vi", 
                            colbert_model="model_trained/colbert/32-384-mask-punct-200.dnn", 
                            index_root="legal_corpus_index/colbert", 
                            index_name="legal_32_384-200-dpr-sep-final", 
                            faiss_partitions=100, 
                            gpu=False)
prf_rank = pytcolbert.end_to_end()
DATA_ROOT = "zalo_data"

from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("src/libs/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2000m')
def word_tokenizer(text):
    new_sent = []
    try:
        sentences = rdrsegmenter.tokenize(text)
    except:
        print(text)
        return text
    for sent in sentences:
        tmp = " ".join(sent)
        new_sent.append(tmp)
    return " ".join(new_sent)

stop_words = []
with open(os.path.join(DATA_ROOT, "stop_words.txt"),"r",encoding="utf-8") as lines:
    for line in lines:
        stop_words.append(line.strip())

def preprocess_model(text):
    try:
        text = re.sub("^Điều [0-9][0-9]*?. "," ",text).strip()
        text = re.sub("^[0-9][0-9]*?. "," ",text).strip()
        return text
    except:
        return ""

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ_]',' ',text)
    return text
    
def load_legal_corpus(path):
    with open(path,'r',encoding="utf-8") as fr:
        legal_corpus = json.load(fr)
    articles_all = []
    article_ids_all = []
    id2article = {}
    for law in legal_corpus:
        for article in law['articles']:
            articles_all.append(re.sub("Điều [0-9][0-9]*?. "," ",article['title']) + " " + article['text'])
            article_ids_all.append(law["law_id"] +"_"+article["article_id"])
            id2article[law["law_id"] +"_"+article["article_id"]] = [article['title'],article['text']]
    return article_ids_all,articles_all,id2article

article_ids_all,articles_all,id2article = load_legal_corpus(os.path.join(DATA_ROOT,"legal_corpus_word_segment.json"))
article_ids_all_old,articles_all_old,id2article_old = load_legal_corpus(os.path.join(DATA_ROOT,"legal_corpus.json"))
article_preprocessed = []
for i in range(len(articles_all)):
    article_preprocessed.append(preprocess(articles_all[i]))
tokenized_corpus = []
for doc in article_preprocessed:
    tmp = []
    for w in doc.split():
        if (not w.isnumeric()) and (w not in stop_words) and len(w)>2:
            tmp.append(w)
    tokenized_corpus.append(tmp[0:512])

bm25 = BM25Plus(tokenized_corpus)
print("build IR BM25 model done! ")

def reranking_bm25(query,ids_list,top_k=10000):
    df_result = prf_rank.search(query).head(top_k)
    top_ids = []
    for idx,rows in df_result.iterrows():
        top_ids.append(rows["docno"])
    relevant_id = []
    for id_ in top_ids:
        if(id_ in ids_list):
            relevant_id.append(id_)
            if(len(relevant_id)>=100):
                break
    return relevant_id

def get_top_n(query_root,top_k=1000):
    query = word_tokenizer(query_root)
    query = preprocess(query)
    tokenized_query = query.split(" ")
    tokenized_query = [w for w in tokenized_query if (not w.isnumeric()) and (w not in stop_words) and len(w)>2]
    top_n_text,top_n_ids = bm25.get_top_n(tokenized_query, articles_all, n=200)
    result_ids = [article_ids_all[i] for i in top_n_ids]
    result_ids = reranking_bm25(query_root,result_ids)[0:top_k]
    result_ids= result_ids[0:top_k]
    top_n_text = [i.split("_")[0]+"[SEP]"+preprocess_model(id2article_old[i][0]) + "[SEP]" + preprocess_model(id2article_old[i][1]) for i in result_ids]
    return result_ids,top_n_text

