import json
import os

def post_processing(result):
    THRESH=0.999
    new_submission = []
    for e in result:
        max_article = {}
        article_list = []
        probs = []
        article2prob = {}
        tmp = {"question_id":e["question_id"],"relevant_articles":[]}
        for article in e["relevant_articles"]:
            article2prob[article["law_id"]+"_"+article["article_id"]] = article["prob"]
        article2prob = dict(sorted(article2prob.items(), key=lambda item: item[1],reverse=True))
        for k in article2prob:
            article_list.append(k)
            probs.append(article2prob[k])

        for i in range(len(article_list)):
            if len(max_article)==0 or probs[i]>max_article["prob"]:
                max_article = {
                                "law_id": article_list[i].split("_")[0],
                                "article_id": article_list[i].split("_")[1],
                                "prob":probs[i]
                            }
            if probs[i] > THRESH:
                tmp["relevant_articles"].append(
                        {
                        "law_id": article_list[i].split("_")[0],
                        "article_id": article_list[i].split("_")[1],
                        "prob":probs[i]
                        }
                )
        tmp["relevant_articles"] = tmp["relevant_articles"][0:1]
        if len(tmp["relevant_articles"])==0:
            if max_article["prob"]>=0:
                tmp["relevant_articles"].append(max_article)
        new_submission.append(tmp)
    return new_submission