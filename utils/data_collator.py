# -*- coding: utf-8 -*-
import torch

def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    if len(list_l) <= max_len:
        padding_l = [padding_value] * (max_len - len(list_l))
        padded_list = list_l + padding_l
    else:
        padded_list = list_l[:max_len]
    return padded_list


class BridgeCollator(object):
    """
    Data collator for brownian bridge mapping.
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0).masked_fill(data_tensor != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_user_utterance_input= []
        batch_delta_follow_input = []
        batch_interim_subgoal_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_transition_input = []
        batch_interim_t, batch_target_T = [], []

        for sample in mini_batch:
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_interim_subgoal_input.append(sample.interim_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_interim_t.append(sample.interim_t)
            batch_target_T.append(sample.target_T)

        # inputs
        user_utt_ids = self.list_to_tensor(batch_user_utterance_input)
        user_utt_mask = self.get_attention_mask(user_utt_ids)

        delta_follow_ids = self.list_to_tensor(batch_delta_follow_input)
        delta_follow_mask = self.get_attention_mask(delta_follow_ids)

        transition_ids = self.list_to_tensor(batch_transition_input)
        transition_mask = self.get_attention_mask(transition_ids)

        interim_subgoal_ids = self.list_to_tensor(batch_interim_subgoal_input)
        interim_subgoal_mask = self.get_attention_mask(interim_subgoal_ids)

        start_subgoal_ids = self.list_to_tensor(batch_start_subgoal_input)
        start_subgoal_mask = self.get_attention_mask(start_subgoal_ids)

        target_subgoal_ids = self.list_to_tensor(batch_target_subgoal_input)
        target_subgoal_mask = self.get_attention_mask(target_subgoal_ids)

        interim_t = torch.tensor(batch_interim_t, dtype=torch.long).to(self.device).contiguous()
        target_T  = torch.tensor(batch_target_T, dtype=torch.long).to(self.device).contiguous()

        collated_batch = {
            "user_utterance": [user_utt_ids, user_utt_mask],
            "delta_follow": [delta_follow_ids, delta_follow_mask],
            "transition": [transition_ids, transition_mask],
            "interim_subgoal": [interim_subgoal_ids, interim_subgoal_mask], 
            "start_subgoal": [start_subgoal_ids, start_subgoal_mask], 
            "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
            "interim_t": interim_t,
            "target_T": target_T,
        }

        return collated_batch


class PlannerCollator(object):
    """
    Data collator for planning.
    """
    def __init__(self, device, model, latent_dim, padding_idx=0, is_eval=False):
        self.device = device
        self.model = model
        self.latent_dim = latent_dim
        self.padding_idx = padding_idx
        self.is_eval = is_eval
    
    def list_to_tensor(self, list_l, special_padding_value=None):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            if special_padding_value is None:
                padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
            else:
                padded_lists.append(pad_sequence(list_seq, max_len, padding_value=special_padding_value))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0).masked_fill(data_tensor != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask


    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_input = []
        batch_decoder_input_all = []
        batch_transition_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_transition_number = []

        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_decoder_input_all.append(sample.decoder_input_all_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_number.append(sample.transition_number)
            
        input_ids = self.list_to_tensor(batch_input)
        input_mask = self.get_attention_mask(input_ids)
        decoder_input_all_ids = self.list_to_tensor(batch_decoder_input_all)
        decoder_input_all_mask = self.get_attention_mask(decoder_input_all_ids)

        start_subgoal_ids = self.list_to_tensor(batch_start_subgoal_input)
        start_subgoal_mask = self.get_attention_mask(start_subgoal_ids)

        target_subgoal_ids = self.list_to_tensor(batch_target_subgoal_input)
        target_subgoal_mask = self.get_attention_mask(target_subgoal_ids)

        user_utt_ids = self.list_to_tensor(batch_user_utterance_input)
        user_utt_mask = self.get_attention_mask(user_utt_ids)

        delta_follow_ids = self.list_to_tensor(batch_delta_follow_input)
        delta_follow_mask = self.get_attention_mask(delta_follow_ids)

        transition_number = torch.tensor(batch_transition_number, dtype=torch.long).to(self.device).contiguous()
        
        batch_tc_mask = []
        for bsz, sample in enumerate(mini_batch):
            tc_mask_temp = (len(sample.decoder_input_ids_list) - 1)  * [1]
            batch_tc_mask.append(tc_mask_temp)
        tc_mask = self.list_to_tensor(batch_tc_mask, special_padding_value=0)
        
        gold_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                    if idx == len(sample.decoder_input_ids_list)-1:
                        continue
                    temp_ids = self.list_to_tensor([dec_ids])
                    temp_mask = self.get_attention_mask(temp_ids)
                    gold_temp[bsz, idx, :] = self.model.get_time_control_embed(temp_ids, temp_mask)
        
        simulate_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                start_latent = self.model.get_time_control_embed(start_subgoal_ids[bsz:bsz+1, :], start_subgoal_mask[bsz:bsz+1, :])
                target_latent = self.model.get_time_control_embed(target_subgoal_ids[bsz:bsz+1, :], target_subgoal_mask[bsz:bsz+1, :])
                Z_u = self.model.get_user_utt_representation(user_utt_ids[bsz:bsz+1, :], user_utt_mask[bsz:bsz+1, :])
                delta_u = self.model.get_delta_u_representation(delta_follow_ids[bsz:bsz+1, :], delta_follow_mask[bsz:bsz+1, :])
                
                # simulate Brownian bridge trjectories
                simulate_bridge_points = self.model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent, T=len(sample.decoder_input_ids_list), Z_u=Z_u, delta_u=delta_u)
                
                assert len(simulate_bridge_points) == len(sample.decoder_input_ids_list)
                for idx, embed in enumerate(simulate_bridge_points[1:]):
                    simulate_temp[bsz, idx, :] = embed
        
        if not self.is_eval:
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(), decoder_input_all_mask[:, :-1].contiguous()],
                "label": [decoder_input_all_ids[:, 1:].contiguous(), decoder_input_all_mask[:, 1:].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],   
            }
        else:
            gold_bridge_list = []
            for bsz, sample in enumerate(mini_batch):
                tc_list = []
                if len(sample.decoder_input_ids_list) > 1:
                    for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                        if idx == len(sample.decoder_input_ids_list)-1:
                            continue
                        temp_ids = self.list_to_tensor([dec_ids])
                        temp_mask = self.get_attention_mask(temp_ids)
                        rep = self.model.get_time_control_embed(temp_ids, temp_mask)
                        tc_list.append(rep)
                gold_bridge_list.append(tc_list)
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(), decoder_input_all_mask[:, :-1].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask], 
                "user_utterance": [user_utt_ids, user_utt_mask],
                "delta_follow": [delta_follow_ids, delta_follow_mask],
                "start_subgoal": [start_subgoal_ids, start_subgoal_mask], 
                "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
                "gold_bridge_list": gold_bridge_list,
            }

        return collated_batch


class DialogCollator(object):
    """
    Data collator for dialogue generation.
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(data_tensor != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_input, batch_label = [], []
        
        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_label.append(sample.lm_labels)
        
        input_ids = self.list_to_tensor(batch_input)
        lm_labels = self.list_to_tensor(batch_label)
        
        collated_batch = {
            "input_ids": input_ids,
            "lm_labels": lm_labels
        }

        return collated_batch
