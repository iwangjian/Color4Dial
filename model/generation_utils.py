# -*- coding: utf-8 -*-
import torch
import copy
from typing import Optional, Callable, List
from transformers import (
    LogitsProcessorList, 
    HammingDiversityLogitsProcessor, 
    NoBadWordsLogitsProcessor, 
    MinLengthLogitsProcessor, 
    PrefixConstrainedLogitsProcessor, 
    ForcedBOSTokenLogitsProcessor, 
    ForcedEOSTokenLogitsProcessor, 
    InfNanRemoveLogitsProcessor, 
    RepetitionPenaltyLogitsProcessor, 
    NoRepeatNGramLogitsProcessor, 
    StoppingCriteriaList, 
    MaxLengthCriteria, 
    MaxTimeCriteria,
)

def _get_logits_processor(
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    bad_words_ids: List[List[int]],
    min_length: int,
    max_length: int,
    eos_token_id: int,
    forced_bos_token_id: int,
    forced_eos_token_id: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    remove_invalid_values: bool,
) -> LogitsProcessorList:
    """
    This mathod returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors

def _get_stopping_criteria(max_length: Optional[int], max_time: Optional[float]) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    if max_time is not None:
        stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
    return stopping_criteria
    
def _validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = copy.deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        print ("You set different `max_length` for stopping criteria and `max_length` parameter", flush=True)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria

def greedy_decoding(
    dataset,
    model,
    inputs,
    tokenizer,
    bridge_points,
    bridge_points_mask,
    sample_number,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    ) -> torch.LongTensor:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = _validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    scores = () if (return_dict_in_generate and output_scores) else None
    
    input_ids = inputs["input"][0]
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    input_ids, input_masks = inputs["input"]
    batch_size = input_ids.shape[0]
    device = input_ids.device
    if dataset == "TGConv":
        if sample_number == 1:
            target_subgoal = inputs["target_subgoal"][0][0, 1:-1].tolist()
            return_list = target_subgoal
            return_tensor = torch.tensor(return_list).unsqueeze(0).contiguous().to(device)
            return return_tensor
    else:
        if sample_number == 0:
            target_subgoal = inputs["target_subgoal"][0][0, 1:-1].tolist()
            return_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[A] Say goodbye [T] NULL")) + target_subgoal
            return_tensor = torch.tensor(return_list).unsqueeze(0).contiguous().to(device)
            return return_tensor
        elif sample_number == 1:
            target_subgoal = inputs["target_subgoal"][0][0, 1:-1].tolist()
            return_list = target_subgoal
            return_tensor = torch.tensor(return_list).unsqueeze(0).contiguous().to(device)
            return return_tensor
                
    if dataset == "TGConv":
        dec_ids = torch.tensor([tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("[T]")]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
    else:
        dec_ids = torch.tensor([tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("[A]")]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
    if bridge_points is not None and bridge_points_mask is not None:
        bridge_embeds = torch.full((bridge_points_mask.shape[0], bridge_points_mask.shape[1], model.latent_dim), 0, dtype=torch.float).to(input_ids.device)
        for bsz_idx in range(bridge_points_mask.shape[0]):
            for time_idx in range(bridge_points_mask.shape[1]):
                if bridge_points_mask[bsz_idx, time_idx] == 1:
                    bridge_embeds[bsz_idx, time_idx, :] = bridge_points[time_idx]
        bridge_mask = bridge_points_mask
    else:
        bridge_embeds = None
        bridge_mask = None

    while True:
        # model forward
        model_out = model(
            input_ids=input_ids, attention_mask=input_masks, 
            decoder_input_ids=dec_ids, decoder_attention_mask=None, 
            bridge_embeds=bridge_embeds, bridge_mask=bridge_mask
        )
        # process logits
        next_token_logits = model_out["lm_logits"][:, -1, :]
        next_tokens_scores = logits_processor(dec_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        dec_ids = torch.cat([dec_ids, next_tokens[:, None]], dim=-1)
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
        if unfinished_sequences.max() == 0 or stopping_criteria(dec_ids, scores):
            break
    
    return dec_ids


def model_decode(model, inputs, tokenizer, bridge_points, bridge_points_mask, sample_number, args):
    dataset = args.dataset
    max_length = args.max_dec_len
    assert max_length > 0
    min_length = args.min_length or 0
    repetition_penalty = args.repetition_penalty or None
    diversity_penalty = args.diversity_penalty or None
    no_repeat_ngram_size = args.no_repeat_ngram_size or None
    bad_words_ids = args.bad_words_ids or None
    remove_invalid_values = args.remove_invalid_values or False

    # get logits processor
    logits_processor = _get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
    )
    # get decoding stopping criteria
    stopping_criteria = _get_stopping_criteria(max_length=max_length, max_time=None)
    
    # apply decoding
    output = greedy_decoding(
        dataset=dataset,
        model=model,
        inputs=inputs,
        tokenizer=tokenizer,
        bridge_points=bridge_points,
        bridge_points_mask=bridge_points_mask,
        sample_number=sample_number,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        max_length=max_length
    )

    return output
