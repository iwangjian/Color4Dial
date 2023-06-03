# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BartTokenizer, GPT2Tokenizer


PAD = "[PAD]"     # consistent with Bert tokenizer
UNK = "[UNK]"     # consistent with Bert tokenizer
SEP = "[SEP]"     # consistent with Bert tokenizer

ACT = "[A]"       # denote an action
TPC = "[T]"       # denote a topic
BOS = "[BOS]"     # begin of sequence
EOS = "[EOS]"     # end of sequence

IGNORE_INDEX = -100    # default in CrossEntropyLoss


def get_tokenizer(config_dir, name="bert"):
    if name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, SEP, PAD]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.convert_tokens_to_ids(PAD),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    elif name == "bart":
        tokenizer = BartTokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "sep_token_id": tokenizer.sep_token_id,
        }
    else:
        tokenizer = BertTokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, BOS, EOS]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
            "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
        }
    return tokenizer, num_added_tokens, special_token_id_dict

def convert_ids_to_tokens(output, tokenizer, lang="en"):
    sentences = []
    for idx in range(output.size(0)):
        decode_tokens = tokenizer.decode(output[idx, :]).split()
        return_tokens = []
        for token in decode_tokens:
            if token == BOS:
                continue
            elif token == EOS or token == PAD:
                break
            elif token.startswith(EOS):
                break
            elif token.endswith(EOS):
                return_tokens.append(token.replace(EOS, ""))
                break
            elif token.endswith("<|endoftext|>"):
                return_tokens.append(token.replace("<|endoftext|>", ""))
                break
            elif token.upper() == "NULL":
                return_tokens.append("NULL")
            else:
                return_tokens.append(token)
        if lang == "zh":
            return_str = "".join(return_tokens)
        else:
            return_str = " ".join(return_tokens)
        sentences.append(return_str)
    return sentences

def get_eval_output(path_str: str, topic_only=False):
    if topic_only:
        try:
            if path_str.startswith(TPC):
                topic = path_str.split(TPC)[1].strip()
            else:
                topic = path_str.split(TPC)[0].strip()
        except IndexError:
            topic = UNK
        return topic
    else:
        # parse dioalog path
        # i.e., [A]act[T]tpc[A]...[T]...
        try:
            action = path_str.split(TPC)[0].split(ACT)[-1].strip()
        except IndexError:
            action = UNK
        try:
            if path_str.startswith(ACT):
                topic = path_str.split(ACT)[1].split(TPC)[-1].strip()
            else:
                topic = path_str.split(ACT)[0].split(TPC)[-1].strip()
        except IndexError:
            topic = UNK
        return (action, topic)

def combine_tokens(output, tokenizer):
    return_sentence=[]
    for batch in range(output.size(0)):
        out_sentence = tokenizer.decode(output[batch, :]).replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "").strip()
        return_sentence.append(out_sentence)
    return return_sentence
