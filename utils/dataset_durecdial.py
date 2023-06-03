# -*- coding: utf-8 -*-
import logging
import os
import json
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import SEP, ACT, TPC, IGNORE_INDEX
from utils.dataset_base import BrownianBridgeInput, PlannerInput, DialogInput


class DuRecdialDataset4Bridge(Dataset):
    """
    Self-defined DuRecdialDataset4Bridge class for brownian bridge mapping.
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
        signature = "durecdial_bridge_{}.pkl".format(self.data_partition)
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
            with open(data_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    sample = json.loads(line.strip())
                    data_sample = {
                        "user_profile": sample["user_profile"],
                        "knowledge": sample["knowledge"],
                        "conversation": sample["conversation"],
                        "target": sample["target"],
                        "action_path": sample["action_path"],
                        "topic_path": sample["topic_path"]
                    }
                    all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                user_utt_ids, follow_ids, start_ids, transition_ids, target_ids = self._parse_input_context(sample)
                plan_path_ids_list = self._parse_plan_path(sample)
                transition_length = len(plan_path_ids_list) - 1
                if transition_length <= 1:
                    continue
                for idx in range(transition_length-1):
                    interim_ids = plan_path_ids_list[idx]
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
            user_utt = sample["conversation"][-1]
            user_utt_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(user_utt) + [self.tokenizer.eos_token]
        else:
            user_utt_tokens = [self.tokenizer.bos_token] + [self.tokenizer.eos_token]
        if len(user_utt_tokens) > self.max_seq_len:
            user_utt_tokens = user_utt_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        user_utt_ids = self.tokenizer.convert_tokens_to_ids(user_utt_tokens)

        # delta: follow discrimination
        profile_tokens = [self.tokenizer.bos_token]
        for k, v in sample["user_profile"].items():
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            profile_tokens = profile_tokens + k_toks + v_toks
        profile_tokens = profile_tokens + [self.tokenizer.sep_token]
        follow_tokens = profile_tokens + user_utt_tokens[1:]
        if len(follow_tokens) > self.max_seq_len:
            follow_tokens = follow_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        follow_ids = self.tokenizer.convert_tokens_to_ids(follow_tokens)

        # S0: domain_knowledge + dialogue history
        kg_tokens = [self.tokenizer.bos_token]
        for kg in sample["knowledge"]:
            kg_tok = self.tokenizer.tokenize(" ".join(kg))
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

        # transition: user profile + S0 + ST
        transition_tokens = profile_tokens + start_tokens[1:-1] + [self.tokenizer.sep_token]
        act_toks = self.tokenizer.tokenize(sample["target"][0])
        tpc_toks = self.tokenizer.tokenize(sample["target"][1])
        target_tokens = [ACT] + act_toks + [TPC] + tpc_toks
        transition_tokens = transition_tokens + target_tokens + [self.tokenizer.eos_token]
        if len(transition_tokens) > self.max_seq_len:
            transition_tokens = transition_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        transition_ids = self.tokenizer.convert_tokens_to_ids(transition_tokens)

        # ST: target
        target_tokens = [self.tokenizer.bos_token] + target_tokens + [self.tokenizer.eos_token]
        if len(target_tokens) > self.max_seq_len:
            target_tokens = target_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        
        return (user_utt_ids, follow_ids, start_ids, transition_ids, target_ids)

    def _parse_plan_path(self, sample: dict):
        # dialog plan path
        plan_path_ids_list = []
        for idx in range(len(sample["action_path"])):
            act_toks = self.tokenizer.tokenize(sample["action_path"][idx])
            tpc_toks = self.tokenizer.tokenize(sample["topic_path"][idx])
            at_ids = self.tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
            plan_path_ids_list.append(at_ids)
        return plan_path_ids_list

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


class DuRecdialDataset4Planning(Dataset):
    """
    Self-defined DuRecdialDataset4Planning class for planning.
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
        signature = "durecdial_plan_{}.pkl".format(self.data_partition)
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
                        "user_profile": sample["user_profile"],
                        "knowledge": sample["knowledge"],
                        "conversation": sample["conversation"],
                        "target": sample["target"],
                        "action_path": sample["action_path"],
                        "topic_path": sample["topic_path"]
                    }
                    all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                inputs = self._parse_input_context(sample)
                feature = PlannerInput(**inputs)
                self.instances.append(feature)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
     
    def _parse_input_context(self, sample: dict):
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
        profile_tokens = [self.tokenizer.bos_token]
        for k, v in sample["user_profile"].items():
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            profile_tokens = profile_tokens + k_toks + v_toks
        profile_tokens = profile_tokens + [self.tokenizer.sep_token]
        follow_tokens = profile_tokens + user_utt_tokens[1:]
        if len(follow_tokens) > self.max_seq_len:
            follow_tokens = follow_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        follow_ids = self.tokenizer.convert_tokens_to_ids(follow_tokens)

        # S0: domain_knowledge + dialogue history
        kg_tokens = [self.tokenizer.bos_token]
        for kg in sample["knowledge"]:
            kg_tok = self.tokenizer.tokenize(" ".join(kg))
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

        # transition: user profile + S0 + ST
        transition_tokens = profile_tokens + start_tokens[1:-1] + [self.tokenizer.sep_token]
        act_toks = self.tokenizer.tokenize(sample["target"][0])
        tpc_toks = self.tokenizer.tokenize(sample["target"][1])
        target_tokens = [ACT] + act_toks + [TPC] + tpc_toks
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
        input_tokens = input_tokens + conv_tokens[:-1] + [self.tokenizer.sep_token] + kg_tokens[1:] + profile_tokens[1:-1] + [self.tokenizer.eos_token]
        if len(input_tokens) > self.max_seq_len:
            input_tokens = input_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # decoder input
        decoder_input_ids = []
        decoder_input_lengths = []
        decoder_input_all = [self.tokenizer.bos_token]
        is_end_flag = False
        
        if len(sample["action_path"]) == 1:
            is_end_flag = True
            for idx in range(len(sample["action_path"])):
                act_toks = self.tokenizer.tokenize(sample["action_path"][idx])
                tpc_toks = self.tokenizer.tokenize(sample["topic_path"][idx])
                at_ids = self.tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
                decoder_input_ids.append(at_ids)
                decoder_input_lengths.append(len(at_ids))
                decoder_input_all = decoder_input_all + [ACT] + act_toks + [TPC] + tpc_toks
        elif len(sample["action_path"]) > 1:
            for idx in range(len(sample["action_path"])-1):
                act_toks = self.tokenizer.tokenize(sample["action_path"][idx])
                tpc_toks = self.tokenizer.tokenize(sample["topic_path"][idx])
                at_ids = self.tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
                decoder_input_ids.append(at_ids)
                decoder_input_lengths.append(len(at_ids))
                decoder_input_all = decoder_input_all + [ACT] + act_toks + [TPC] + tpc_toks
        else:
            raise ValueError("action path is empty")
        decoder_input_all = decoder_input_all + [self.tokenizer.eos_token]
        decoder_input_all_ids = self.tokenizer.convert_tokens_to_ids(decoder_input_all)
        assert len(decoder_input_all_ids) == (sum(decoder_input_lengths)+2)

        if is_end_flag:
            transition_number = 0
        else:
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

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


class DuRecdialDataset4Dialog(Dataset):
    """
    Self-defined DuRecdialDataset4Dialog class for dialogue generation.
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
        lower_case=False
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
        self._cache_instances(data_path, plan_path)

    def _cache_instances(self, data_path, plan_path=None):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "durecdial_dialog_{}.pkl".format(self.data_partition)
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
                
                logging.info("Loading planned paths from {}".format(plan_path))
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
                            "user_profile": sample["user_profile"],
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
                        plan_path = self._get_plan_path(
                            sample["action_path"], sample["topic_path"],
                            sample["target"][0], sample["target"][1],
                            lower_case=self.lower_case
                        )   
                        data_sample = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "plan_path": plan_path,
                            "response": sample["response"]
                        }
                        all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                inputs = self._parse_sample(sample)
                feature = DialogInput(**inputs)
                self.instances.append(feature)
            if not self.is_test:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    
    @staticmethod
    def _get_plan_path(action_path: list, topic_path: list, target_action: str, target_topic: str, lower_case: bool=False):
        ptr = -1
        for idx in range(len(action_path)):
            if action_path[idx] == target_action and topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            action_path = action_path[:ptr+1]
            topic_path = topic_path[:ptr+1]
        elif ptr == 0:
            action_path = [action_path[0]]
            topic_path = [topic_path[0]]
        else:
            action_path = action_path + [target_action]
            topic_path = topic_path + [target_topic]
        path_str = ""
        for a, t in zip(action_path, topic_path):
            if lower_case:
                a = a.lower()
                t = t.lower()
            if not t in path_str:
                path_str += "%s %s %s %s " % (ACT, a, TPC, t)
            elif not a in path_str:
                path_str += "%s %s %s %s " % (ACT, a, TPC, t)
        return path_str.strip()

    def _parse_sample(self, sample: dict):
        bos_token_id = self.special_tokens_dict["bos_token_id"]
        eos_token_id = self.special_tokens_dict["eos_token_id"]
        
        # user profile
        profile_tokens = []
        for k, v in sample["user_profile"].items():
            if self.lower_case:
                k = k.lower()
                v = v.lower()
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            profile_tokens +=  k_toks + v_toks + [SEP]

        # domain knowledge
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
        ctx_tokens = profile_tokens + kg_tokens + conv_tokens + plan_tokens
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
  
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]