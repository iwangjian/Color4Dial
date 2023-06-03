# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2LMHeadModel


class DialogModel(nn.Module):
    """
    Model class: DialogModel
    Args:
        args: All necessary arguments for the model.
    """
    def __init__(self, args):
        super().__init__()
        self.model_name = args.base_model
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        self.bos_token_id = args.bos_token_id
        self.eos_token_id = args.eos_token_id

        if args.base_model == "DialoGPT":
            # auto load from https://huggingface.co/microsoft/DialoGPT-small
            self.lm_decoder = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        else:
            # auto load from https://huggingface.co/gpt2
            self.lm_decoder = GPT2LMHeadModel.from_pretrained("gpt2")

        self.lm_decoder.resize_token_embeddings(args.vocab_size)
    
    def compute_acc(self, lm_logits, seq_masks, seq_labels):
        pred = torch.softmax(lm_logits, -1)
        _, pred_y = pred.max(-1)
        hit_tokens = (torch.eq(pred_y, seq_labels)*seq_masks).sum().item()
        num_tokens = seq_masks.sum().item()
        acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
        return acc

    def forward(self, batch, is_test=False):
        if is_test:
            input_ids = batch["input_ids"]

            lm_output = self.lm_decoder(input_ids=input_ids, return_dict=True)
            lm_logits = lm_output["logits"]
            
            output = {"logits": lm_logits}
        else:
            input_ids = batch["input_ids"]
            lm_labels = batch["lm_labels"]
            lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100)
            label_masks = lm_labels.masked_fill(lm_labels == -100, 0)
            label_masks = label_masks.masked_fill(label_masks > 0, 1)

            lm_output = self.lm_decoder(input_ids=input_ids, labels=lm_labels, return_dict=True)
            lm_logits = lm_output["logits"]
            lm_loss = lm_output["loss"]
            
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            shift_masks = label_masks[:, 1:].contiguous()
            acc = self.compute_acc(shift_logits, shift_masks, shift_labels)

            output = {
                "loss": lm_loss,
                "lm_loss": lm_loss,
                "acc": acc
            }
        
        return output


    def generate(self, args, inputs):
        """model inference"""
        n_ctx = self.lm_decoder.config.n_ctx
        special_tokens_ids = [self.pad_token_id, self.eos_token_id]
        max_dec_len = args.max_dec_len
        assert max_dec_len > 0

        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        output_ids = input_ids.new(batch_size, max_dec_len).fill_(self.pad_token_id)

        for batch_idx in range(batch_size):
            idx_inputs = {
                "input_ids": inputs["input_ids"][batch_idx:batch_idx+1],
            }
            cur_input_ids = idx_inputs["input_ids"]
            for len_idx in range(max_dec_len):
                cur_input_ids = cur_input_ids[:, -(n_ctx - 1):]  # (1, seq_len)
                idx_inputs["input_ids"] = cur_input_ids

                lm_output = self.forward(idx_inputs, is_test=True)
                
                logits = lm_output["logits"]
                logits = logits[0, -1, :] / args.temperature
                if args.top_k > 0 or (args.top_p > 0 and args.top_p <= 1):
                    filtered_logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.topk(probs, 1)[1]
                
                if len_idx < args.min_length and next_token.item() in special_tokens_ids:
                    next_token = torch.multinomial(probs, num_samples=1)

                output_ids[batch_idx, len_idx] = next_token
                cur_input_ids = torch.cat([cur_input_ids, next_token.unsqueeze(0)], dim=1)

                if next_token.item() in special_tokens_ids:
                    break

        output_dict = {"response": output_ids}

        return output_dict


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits