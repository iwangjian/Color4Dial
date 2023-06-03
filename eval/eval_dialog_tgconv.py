#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import nltk
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_data(eval_fp, gold_fp):
    all_eval, all_gold = [], []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            all_eval.append(sample)
    with open(gold_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            raw_sample = json.loads(line)
            sample = {
                "id": raw_sample["id"],
                "target": raw_sample["target"],
                "topic_path": raw_sample["topic_path"],
                "conversation": raw_sample["conversation"],
                "response": raw_sample["response"]
            }
            all_gold.append(sample)
    assert len(all_eval) == len(all_gold)
    return (all_eval, all_gold)

def get_eval_response(idx, eval_samples, gold_samples):
    eval_list = [eval_samples[idx]["response"]]
    dialog_id = gold_samples[idx]["id"]
    j = 1
    while j < 8:
        # eval within 8 turns
        if idx - j >= 0 and gold_samples[idx-1]["id"] == dialog_id:
            eval_list.append(eval_samples[idx-1]["response"])
            j += 1
        else:
            break
    return eval_list

def is_topic_hit(topic, candidates):
    for cand in candidates:
        if topic.lower() in cand.lower():
            return True
    return False

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_dialog_similarity(tokenizer, model, all_dialogs):
    all_cos = []
    per_cos = []
    with torch.no_grad():
        for dialog in tqdm(all_dialogs):
            encoded_input = tokenizer(dialog, padding=True, truncation=True, return_tensors="pt").to('cuda')
            model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            dialog_cos = []
            window = 1
            for i in range(window, len(sentence_embeddings)):
                cos = torch.cosine_similarity(sentence_embeddings[i], sentence_embeddings[i-window], dim=0).item()
                dialog_cos.append(cos)
            all_cos.extend(dialog_cos)
            per_cos.append(np.mean(dialog_cos))

    return (np.mean(all_cos), np.mean(per_cos))

def calc_coherence(eval_fp, gold_fp):
    """ Calculate coherence score """
    print("Loading sentence transformer model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')
    model.eval()

    all_eval, all_gold = load_data(eval_fp, gold_fp)
    all_dialogs = []
    current_dialog = []
    for idx in range(1, len(all_gold)):
        if idx == 1:
            current_dialog.append(all_gold[0]["conversation"])
            current_dialog.append(all_eval[0]["response"])

        if all_gold[idx]["id"] == all_gold[idx-1]["id"]:
            current_dialog.append(all_eval[idx]["response"])
        else:
            all_dialogs.append(current_dialog)
            current_dialog = []
            current_dialog.append(all_gold[idx]["conversation"])
            current_dialog.append(all_eval[idx]["response"])
    all_dialogs.append(current_dialog)

    micro_avg_cos, macro_avg_cos = get_dialog_similarity(tokenizer, model, all_dialogs)
    print("Micro Coh.: {:.2f}".format(micro_avg_cos))
    print("Macro Coh.: {:.2f}".format(macro_avg_cos))


def calc_succ(eval_fp, gold_fp):
    """ Calculate goal success rate """
    all_eval, all_gold = load_data(eval_fp, gold_fp)
    topic_hit, topic_total = 0, 0
    
    for idx, gold_sample in enumerate(all_gold):
        if gold_sample["topic_path"][0].lower() == gold_sample["target"].lower() and \
            gold_sample["target"].lower() in gold_sample["response"].lower():
            topic_total += 1
            eval_topic = gold_sample["target"]
            eval_list = get_eval_response(idx, all_eval, all_gold)
            eval_topic = " ".join(nltk.word_tokenize(eval_topic))
            eval_list = [" ".join(nltk.word_tokenize(eval_response)) for eval_response in eval_list]
            if is_topic_hit(eval_topic, eval_list):
                topic_hit += 1
    
    succ_rate = float(topic_hit) / topic_total
    print("Succ.: {}/{} = {:.2f}%".format(topic_hit, topic_total, succ_rate*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    calc_succ(eval_fp=args.eval_file, gold_fp=args.gold_file)
    calc_coherence(eval_fp=args.eval_file, gold_fp=args.gold_file)
