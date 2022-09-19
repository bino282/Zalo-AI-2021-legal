import time
s_time = time.time()
from text_search import *
from reranker import RerankerForInference
from tqdm import tqdm
from postprocess_result import *
import torch
from torch.utils.data import TensorDataset,SequentialSampler,DataLoader
rk = RerankerForInference.from_pretrained("model_trained/rerank/colbert-200-100/") # load checkpoint
print("total time for load_model:{}".format(time.time()-s_time))
rk.hf_model.cuda()
def predict(questions,passages):
    scores = []
    inputs = rk.tokenize([(questions[i], passages[i]) for i in range(len(questions))],max_length=512,padding="max_length",truncation="only_second",return_token_type_ids=True,return_tensors='pt')
    all_input_ids = inputs["input_ids"]
    all_attention_mask = inputs["attention_mask"]
    all_token_type_ids = inputs["token_type_ids"]
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    data_sampler = SequentialSampler(dataset)
    batch_size = 4
    dataloader = DataLoader(dataset=dataset,
                                sampler=data_sampler,
                                batch_size=batch_size,
                                num_workers=1)
    all_score = []
    for idx, batch in enumerate(dataloader):
        input_ids, attention_mask, token_type_ids = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        batch = {
            "input_ids":input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        scores = rk(batch).logits
        scores = torch.sigmoid(scores)
        scores= scores.cpu().detach().numpy().flatten().tolist()
        all_score = all_score + scores
    return all_score

def main():
    DATA_ROOT = "zalo_data/"
    MODE = "test"
    if (MODE=="dev"):
        path_in = "dev_split_v2.json"
        path_out = "test_ann.json"
    else:
        list_file_in = os.listdir("/data")
        if len(list_file_in)!=1:
            print("There is only 1 test json file in the data folder. Found {} file".format(len(list_file_in)))
            exit()
        path_in = os.path.join("/data",list_file_in[0])
        if not os.path.exists("/result"):
            os.mkdir("/result")
        path_out = "/result/submission.json"
    pred_items = []
    with open(path_in,'r',encoding="utf-8") as fr:
        test_data = json.load(fr)["items"]
    print("start predict on test data")
    s_time = time.time()
    for item in tqdm(test_data):
        question = item["question"]
        top_n_ids,top_n_text = get_top_n(question,top_k=50)
        tmp_relevant = []
        probs = predict([question]*len(top_n_ids),top_n_text)
        for i in range(len(probs)):
            if probs[i]>0.0:
                law_id = top_n_ids[i].split("_")[0]
                article_id = top_n_ids[i].split("_")[1]
                tmp_relevant.append({"law_id":law_id,"article_id":article_id,"prob":probs[i]})
        if (len(tmp_relevant)==0):
            for k in range(len(top_n_ids[0:3])):
                doc_id = top_n_ids[k]
                tmp_relevant.append({
                    "law_id":doc_id.split("_")[0],
                    "article_id":doc_id.split("_")[1],
                    "prob":1
                })
        pred_items.append({
            "question_id": item["question_id"],
            "relevant_articles":tmp_relevant
        })
    if(MODE=="test"):
        test_ann = pred_items
    else:
        test_ann = {
            "_name_": "test_ann",
            "_count_": len(pred_items),
            "items": pred_items
        }
    test_ann = post_processing(test_ann)
    with open(path_out,"w",encoding="utf-8") as fw:
        json.dump(test_ann,fw,indent=4,ensure_ascii=False)
    print("End prediction!")
    print("total time for predict on test : {}".format(time.time() - s_time))
    rdrsegmenter.close()

if __name__ == "__main__":
    main()