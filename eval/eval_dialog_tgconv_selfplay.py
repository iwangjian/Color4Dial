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


def load_data(eval_fp):
    all_eval = []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            dialog = json.loads(line)
            all_eval.append(dialog)
    return all_eval

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

def calc_coherence(eval_fp):
    """ Calculate coherence score """
    print("Loading sentence transformer model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')
    model.eval()

    all_eval = load_data(eval_fp)
    all_dialogs = []

    for dialog in all_eval:
        current_dialog = dialog["init_conversation"]
        current_dialog.extend(dialog["generated_response_list"])
    all_dialogs.append(current_dialog)

    micro_avg_cos, macro_avg_cos = get_dialog_similarity(tokenizer, model, all_dialogs)
    print("Micro Coh.: {:.2f}".format(micro_avg_cos))
    print("Macro Coh.: {:.2f}".format(macro_avg_cos))


def calc_succ(eval_fp):
    """ Calculate goal success rate """
    all_eval = load_data(eval_fp)
    topic_hit, topic_total = 0, 0
    
    for eval_dialog in all_eval:
        eval_topic = eval_dialog["target"]
        topic_total += 1
        eval_list = eval_dialog["generated_response_list"]
        eval_topic = " ".join(nltk.word_tokenize(eval_topic))
        eval_list = [" ".join(nltk.word_tokenize(eval_response)) for eval_response in eval_list]
        if is_topic_hit(eval_topic, eval_list):
            topic_hit += 1
    
    succ_rate = float(topic_hit) / topic_total
    print("Succ.: {}/{} = {:.2f}%".format(topic_hit, topic_total, succ_rate*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    args = parser.parse_args()

    calc_succ(eval_fp=args.eval_file)
    calc_coherence(eval_fp=args.eval_file)
