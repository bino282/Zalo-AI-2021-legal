from rank_bm25 import *
import os
import re
import json
from tqdm import tqdm
import pandas as pd
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("/home/ntq/nhanv/git/legal-text-retrieval/libs/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx4000m')

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

DATA_ROOT = "zalo_data"
stop_words = []
with open("stop_words.txt","r",encoding="utf-8") as lines:
    for line in lines:
        stop_words.append(line.strip())

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
    for law in tqdm(legal_corpus):
        for article in law['articles']:
            articles_all.append(re.sub("Điều [0-9][0-9]*?. "," ",article['title']) + " " + article['text'])
            article_ids_all.append(law["law_id"] +"_"+article["article_id"])
            id2article[law["law_id"] +"_"+article["article_id"]] = [article["title"],article['text']]

    return article_ids_all,articles_all,id2article

article_ids_all,articles_all,id2article = load_legal_corpus(os.path.join(DATA_ROOT,"legal_corpus_word_segment.json"))
article_ids_all_root,articles_all_root,id2article_root = load_legal_corpus(os.path.join(DATA_ROOT,"legal_corpus.json"))
article_preprocessed = []
print("preprocessing.....")
for i in tqdm(range(len(articles_all))):
    article_preprocessed.append(preprocess(articles_all[i]))
tokenized_corpus = []
for doc in article_preprocessed:
    tmp = []
    for w in doc.split():
        if (not w.isnumeric()) and (w not in stop_words) and len(w)>2:
            tmp.append(w)
    tokenized_corpus.append(tmp[0:512])
bm25 = BM25Plus(tokenized_corpus,k1=1.5, b=0.75)

with open(os.path.join(DATA_ROOT,"train_split_v2.json"),"r",encoding="utf-8") as fr:
    dev_split_data = json.load(fr)

true_item = 0
total_item = 0
labels = []
queries = []
articles = []
titles= []
artilce_ids = []

items_predict = []

for item in tqdm(dev_split_data["items"]):
    query = item["question"]
    query = word_tokenizer(query)
    query = preprocess(query)
    tokenized_query = query.split(" ")
    tokenized_query = [w for w in tokenized_query if (not w.isnumeric()) and (w not in stop_words) and(len(w)>2)]
    top_n_text,top_n_ids = bm25.get_top_n(tokenized_query, article_preprocessed, n=1000)
    result_ids = [article_ids_all[i] for i in top_n_ids]

    gold_article_ids = []
    for article in item["relevant_articles"]:
        queries.append(item["question"])
        labels.append('1')
        article_id = article["law_id"] + "_" + article["article_id"]
        gold_article_ids.append(article_id)
        articles.append(id2article_root[article_id][1])
        titles.append(id2article_root[article_id][0])
        artilce_ids.append(article["law_id"] + "_" + article["article_id"])
    for article_id in result_ids:
        if article_id not in gold_article_ids:
            queries.append(item["question"])
            labels.append('0')
            articles.append(id2article_root[article_id][1])
            titles.append(id2article_root[article_id][0])
            artilce_ids.append(article_id)

    for article in item["relevant_articles"]:
        article_id = article["law_id"] + "_" + article["article_id"]
        if article_id in result_ids:
            true_item = true_item + 1
        total_item = total_item + 1

rdrsegmenter.close()
df = pd.DataFrame({"label": labels,"query": queries,"article": articles,"title":titles,"id":artilce_ids})
df.to_csv(os.path.join(DATA_ROOT,"train_pairs_1000.csv"),index=False)
print(true_item/total_item)
