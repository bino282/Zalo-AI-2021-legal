# Zalo AI Challeng 2021

## Legal Retrieval

Follow train system : 

    - Train DPR, get top 200 document for a query

    - Use data from DPR train ColBERT

    - Use ColBERT get top 100 for a query

    - Traing reranking from data output ColBERT

Follow inference system:
    
    - Get top 200 documents for a query from BM25
    - Use ColBERT get top 50 documents for a query from BM25 result
    - Reranking 50 document for a query from ColBERT result
    - GET top 1 documents for a query from Reranking result

# Dense Passage Retrieval (src/GC-DPR)

## Retriever input data
The preprocessed data available at src/GC-DPR/data or you can create by run file : create_training_data_dpr.py (create from top 1000 BM25(view src/bm25/test.py))

## Training and inference

For training: ./train_dense.sh

For encode legal corpus: ./encode_corpus.sh

For serving: serving.py (http://localhost:9555/retrieval)

# Dense Passage Retrieval (src/ColBERT)

## Retriever input data
The preprocessed data code at src/create_colbert_training_data.py or you can create by run file : create_colbert_training_data.py

## Training

./train.sh

# ReRank (src/Rerank)

Create data from colbert : create_top100_train_from_colbert.py

Build train data for ranking: ./build_train_from_ranking.sh

Training : ./train.sh
