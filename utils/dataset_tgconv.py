# -*- coding: utf-8 -*-
import logging
import os
import json
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import SEP, TPC, IGNORE_INDEX
from utils.dataset_base import BrownianBridgeInput, PlannerInput, DialogInput


class TGConvDataset4Bridge(Dataset):
    """
    Self-defined TGConvDataset4Bridge class for brownian bridge mapping.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path,
        data_partition,
        tokenizer,
        cache_dir=None,
        max_seq_len=512,
        turn_type_size=16,
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.turn_type_size = turn_type_size
        
        self.instances = []
        self._cache_instances(data_path)
    
    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "tgconv_bridge_{}.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info("Loading raw data from {}".format(data_path))
            all_samples = []
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    sample = json.loads(line.strip())
                    data_sample = {
                        "knowledge": sample["knowledge"],
                        "conversation": sample["conversation"],
                        "target": sample["target"],
                        "topic_path": sample["topic_path"],
                        "response": sample["response"],
                        "hard_target": sample["hard_target"],
                    }
                    all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                user_utt_ids, follow_ids, start_ids, transition_ids, target_ids = self._parse_input_context(sample)
                plan_path_ids = self._parse_plan_path(sample)
                transition_length = len(plan_path_ids)
                if transition_length <= 1:
                    continue
                for idx in range(transition_length-1):
                    interim_ids = plan_path_ids[idx]
                    inputs = {
                        "user_utt_ids": user_utt_ids,
                        "follow_ids": follow_ids,
                        "transition_ids": transition_ids,
                        "interim_ids": interim_ids,
                        "start_ids": start_ids,
                        "target_ids": target_ids,
                        "interim_t": idx + 1,
                        "target_T": transition_length,
                    }
                    feature = BrownianBridgeInput(**inputs)
                    self.instances.append(feature)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
     
    def _parse_input_context(self, sample: dict):
        # last user utterance
        if len(sample["conversation"]) > 0:
            user_utt = sample['conversation'][-1]
            user_utt_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(user_utt) + [self.tokenizer.eos_token]
        else:
            user_utt_tokens = [self.tokenizer.bos_token] + [self.tokenizer.eos_token]
        if len(user_utt_tokens) > self.max_seq_len:
            user_utt_tokens = user_utt_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        user_utt_ids = self.tokenizer.convert_tokens_to_ids(user_utt_tokens)

        # delta: follow discrimination
        follow_tokens = [self.tokenizer.bos_token] + user_utt_tokens[1:]
        if len(follow_tokens) > self.max_seq_len:
            follow_tokens = follow_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        follow_ids = self.tokenizer.convert_tokens_to_ids(follow_tokens)

        # S0: domain_knowledge + dialogue history
        kg_tokens = [self.tokenizer.bos_token]
        for kg in sample["knowledge"]:
            s, p, o = kg
            kg_tok = self.tokenizer.tokenize(" ".join([s, o]))
            kg_tokens = kg_tokens + kg_tok + [self.tokenizer.sep_token]
        conv_tokens = []
        history = sample["conversation"]
        if len(history) > self.turn_type_size:
            history = history[-self.turn_type_size:]
        for h in history:
            h_toks = self.tokenizer.tokenize(h)
            conv_tokens = conv_tokens + h_toks
        conv_tokens = conv_tokens + [self.tokenizer.eos_token]
        start_tokens = kg_tokens + conv_tokens
        if len(start_tokens) > self.max_seq_len:
            start_tokens = start_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        start_ids = self.tokenizer.convert_tokens_to_ids(start_tokens)

        # transition: S0 + ST
        transition_tokens = [self.tokenizer.bos_token] + start_tokens[1:-1] + [self.tokenizer.sep_token]
        tpc_toks = self.tokenizer.tokenize(sample["target"])
        target_tokens = [TPC] + tpc_toks
        transition_tokens = transition_tokens + target_tokens + [self.tokenizer.eos_token]
        if len(transition_tokens) > self.max_seq_len:
            transition_tokens = transition_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        transition_ids = self.tokenizer.convert_tokens_to_ids(transition_tokens)

        # target
        target_tokens = [self.tokenizer.bos_token] + target_tokens + [self.tokenizer.eos_token]
        if len(target_tokens) > self.max_seq_len:
            target_tokens = target_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        
        return (user_utt_ids, follow_ids, start_ids, transition_ids, target_ids)

    def _parse_plan_path(self, sample: dict):
        plan_path_ids = []
        for idx in range(len(sample["topic_path"])):
            tpc_toks = self.tokenizer.tokenize(sample["topic_path"][idx])
            tpc_ids = self.tokenizer.convert_tokens_to_ids([TPC] + tpc_toks)
            plan_path_ids.append(tpc_ids)   
        return plan_path_ids

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


class TGConvDataset4Planning(Dataset):
    """
    Self-defined TGConvDataset4Planning class for planning.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path,
        data_partition,
        tokenizer,
        cache_dir=None,
        max_seq_len=512,
        turn_type_size=16,
        selfplay_mode=None
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.turn_type_size = turn_type_size
        
        self.instances = []
        self._cache_instances(data_path, selfplay_mode=selfplay_mode)
    
    def _cache_instances(self, data_path, selfplay_mode=None):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        if selfplay_mode is not None:
            logging.info("Loading raw data from {}".format(data_path))
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    sample = json.loads(line.strip())
                    if selfplay_mode == "easy":
                        target = sample["target"]
                        knowledge = sample["knowledge"]
                    elif selfplay_mode == "hard":
                        target = sample["hard_target"]
                        knowledge = sample["hard_knowledge"]
                    else:
                        raise ValueError("Unknown mode: {}".format(selfplay_mode))
                    data_sample = {
                        "id": sample["id"],
                        "knowledge": knowledge,
                        "conversation": sample["conversation"],
                        "target": target,
                        "topic_path": sample["topic_path"],
                        "response": sample["response"],
                    }
                    self.instances.append(data_sample)
        else:
            signature = "tgconv_plan_{}.pkl".format(self.data_partition)
            if self.cache_dir is not None:
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                cache_path = os.path.join(self.cache_dir, signature)
            else:
                cache_dir = os.mkdir("caches")
                cache_path = os.path.join(cache_dir, signature)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    logging.info("Loading cached instances from {}".format(cache_path))
                    self.instances = pickle.load(f)
            else:          
                logging.info("Loading raw data from {}".format(data_path))
                all_samples = []
                with open(data_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        sample = json.loads(line.strip())
                        data_sample = {
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"],
                            "topic_path": sample["topic_path"]
                        }
                        all_samples.append(data_sample)
                
                logging.info("Creating cache instances {}".format(signature))
                for sample in tqdm(all_samples):
                    inputs = self.parse_input_context(sample)
                    feature = PlannerInput(**inputs)
                    self.instances.append(feature)
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.instances, f)
            logging.info("Total of {} instances were cached.".format(len(self.instances)))
     
    def parse_input_context(self, sample: dict):
        # last user utterance
        if len(sample["conversation"]) > 0:
            user_utt = sample["conversation"][-1]
            user_utt_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(user_utt) + [self.tokenizer.eos_token]
        else:
            user_utt_tokens = [self.tokenizer.bos_token] + [self.tokenizer.eos_token]
        if len(user_utt_tokens) > self.max_seq_len:
            user_utt_tokens = user_utt_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        user_utt_ids = self.tokenizer.convert_tokens_to_ids(user_utt_tokens)

        # delta: follow discrimination
        follow_tokens = [self.tokenizer.bos_token] + user_utt_tokens[1:]
        if len(follow_tokens) > self.max_seq_len:
            follow_tokens = follow_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        follow_ids = self.tokenizer.convert_tokens_to_ids(follow_tokens)

        # S0: domain_knowledge + dialogue history
        kg_tokens = [self.tokenizer.bos_token]
        for kg in sample["knowledge"]:
            s, p, o = kg
            kg_tok = self.tokenizer.tokenize(" ".join([s, o]))
            kg_tokens = kg_tokens + kg_tok + [self.tokenizer.sep_token]
        conv_tokens = []
        history = sample["conversation"]
        if len(history) > self.turn_type_size:
            history = history[-self.turn_type_size:]
        for h in history:
            h_toks = self.tokenizer.tokenize(h)
            conv_tokens = conv_tokens + h_toks
        conv_tokens = conv_tokens + [self.tokenizer.eos_token]
        start_tokens = kg_tokens + conv_tokens
        if len(start_tokens) > self.max_seq_len:
            start_tokens = start_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        start_ids = self.tokenizer.convert_tokens_to_ids(start_tokens)

        # transition: S0 + ST
        transition_tokens = start_tokens[:-1] + [self.tokenizer.sep_token]
        tpc_toks = self.tokenizer.tokenize(sample["target"])
        target_tokens = [TPC] + tpc_toks
        transition_tokens = transition_tokens + target_tokens + [self.tokenizer.eos_token]
        if len(transition_tokens) > self.max_seq_len:
            transition_tokens = transition_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        transition_ids = self.tokenizer.convert_tokens_to_ids(transition_tokens)

        # ST: target
        target_tokens = [self.tokenizer.bos_token] + target_tokens + [self.tokenizer.eos_token]
        if len(target_tokens) > self.max_seq_len:
            target_tokens = target_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)

        # input
        input_tokens = [self.tokenizer.bos_token]
        input_tokens = input_tokens + conv_tokens[:-1] + [self.tokenizer.sep_token] + kg_tokens[1:] + [self.tokenizer.eos_token]
        if len(input_tokens) > self.max_seq_len:
            input_tokens = input_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # decoder input
        decoder_input_ids = []
        decoder_input_lengths = []
        decoder_input_all = [self.tokenizer.bos_token]
        for idx in range(len(sample["topic_path"])):
            tpc_toks = self.tokenizer.tokenize(sample["topic_path"][idx])
            at_ids = self.tokenizer.convert_tokens_to_ids([TPC] + tpc_toks)
            decoder_input_ids.append(at_ids)
            decoder_input_lengths.append(len(at_ids))
            decoder_input_all = decoder_input_all + [TPC] + tpc_toks
        
        decoder_input_all = decoder_input_all + [self.tokenizer.eos_token]
        decoder_input_all_ids = self.tokenizer.convert_tokens_to_ids(decoder_input_all)
        assert len(decoder_input_all_ids) == (sum(decoder_input_lengths)+2)

        transition_number = len(decoder_input_ids)
        inputs = {
            "input_ids": input_ids,
            "decoder_input_ids_list": decoder_input_ids,
            "decoder_input_all_ids": decoder_input_all_ids,
            "transition_ids": transition_ids,
            "start_ids": start_ids,
            "target_ids": target_ids,
            "user_utt_ids": user_utt_ids,
            "follow_ids": follow_ids,
            "transition_number": transition_number,
        }

        return inputs

    def get_items(self):
        return self.instances
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


class TGConvDataset4Dialog(Dataset):
    """
    Self-defined TGConvDataset4Dialog class for dialogue generation.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, 
        data_path,  
        data_partition,
        tokenizer,
        special_tokens_dict,
        cache_dir=None, 
        max_seq_len=512,
        turn_type_size=16,
        is_test=False,
        plan_path=None,
        lower_case=False,
        selfplay_mode=None
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.special_tokens_dict = special_tokens_dict
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.turn_type_size = turn_type_size
        self.is_test = is_test
        self.lower_case = lower_case
        
        self.instances = []
        if selfplay_mode is None:
            self._cache_instances(data_path, plan_path)

    def _cache_instances(self, data_path, plan_path=None):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "tgconv_dialog_{}.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            if self.is_test:
                if plan_path is None:
                    raise ValueError("`plan_path` should not be None during inference!")
                
                logging.info("Loading raw data from {}".format(plan_path))
                all_plans = []
                with open(plan_path, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        plan = json.loads(line.strip())
                        all_plans.append(plan)
                
                logging.info("Loading raw data from {}".format(data_path))
                all_samples = []
                with open(data_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        sample = json.loads(line.strip())
                        data_sample = {
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"]
                        }
                        all_samples.append(data_sample)
                
                assert len(all_samples) == len(all_plans)
                for sample, plan in zip(all_samples, all_plans):
                    sample["plan_path"] = plan["plan_path"]
            
            else:
                logging.info("Loading raw data from {}".format(data_path))
                all_samples = []
                with open(data_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        sample = json.loads(line.strip())
                        plan_path = self.get_plan_path(
                            sample["topic_path"], sample["target"],
                            lower_case=self.lower_case
                        )   
                        data_sample = {
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "plan_path": plan_path,
                            "response": sample["response"]
                        }
                        all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                inputs = self.parse_sample(sample)
                feature = DialogInput(**inputs)
                self.instances.append(feature)
            if not self.is_test:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    
    @staticmethod
    def get_plan_path(topic_path: list, target_topic: str, lower_case: bool=False):
        ptr = -1
        for idx in range(len(topic_path)):
            if topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            topic_path = topic_path[:ptr+1]
        elif ptr == 0:
            topic_path = [topic_path[0]]
        else:
            topic_path = topic_path + [target_topic]
        path_str = ""
        for t in topic_path:
            t = t.lower() if lower_case else t
            path_str += "%s %s " % (TPC, t)
        return path_str.strip()

    def parse_sample(self, sample: dict):
        bos_token_id = self.special_tokens_dict["bos_token_id"]
        eos_token_id = self.special_tokens_dict["eos_token_id"]

        # commonsense knowledge
        kg_tokens = []
        for kg in sample["knowledge"]:
            kg_str = " ".join(kg)
            if self.lower_case:
                kg_str = kg_str.lower()
            kg_toks = self.tokenizer.tokenize(kg_str)
            kg_tokens +=  kg_toks + [SEP]

        # dialogue history
        conv_tokens = []
        history = sample["conversation"]
        if len(history) > self.turn_type_size:
            history = history[-self.turn_type_size:]
        for h in history:
            if self.lower_case:
                h = h.lower()
            h_toks = self.tokenizer.tokenize(h)
            conv_tokens += h_toks + [SEP]
        
        # plan path
        plan_path = sample["plan_path"]
        plan_tokens = self.tokenizer.tokenize(plan_path)

        # concat as context
        ctx_tokens =  kg_tokens + conv_tokens + plan_tokens
        ctx_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
        if len(ctx_ids) > self.max_seq_len - 1:
            ctx_tokens = ctx_tokens[-self.max_seq_len+1:]
            ctx_ids = ctx_ids[-self.max_seq_len+1:]

        if self.is_test:    
            input_ids = [bos_token_id] + ctx_ids
            lm_labels = [IGNORE_INDEX] * (len(ctx_ids) + 1)
        else:
            resp_str = sample["response"].lower() if self.lower_case else sample["response"]
            resp_tokens = self.tokenizer.tokenize(resp_str)
            resp_ids = self.tokenizer.convert_tokens_to_ids(resp_tokens) + [eos_token_id]
            input_ids = [bos_token_id] + ctx_ids + resp_ids
            lm_labels = [IGNORE_INDEX] * (len(ctx_ids) + 1) + resp_ids
        
        inputs = {
            "input_ids": input_ids,
            "lm_labels": lm_labels
        }
        return inputs
    
    def get_items(self):
        return self.instances
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]
