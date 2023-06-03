# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import json
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from utils.data_utils import get_tokenizer, combine_tokens, convert_ids_to_tokens
from utils.dataset_base import PlannerInput, DialogInput
from utils.dataset_durecdial import DuRecdialDataset4Dialog
from utils.dataset_tgconv import TGConvDataset4Planning, TGConvDataset4Dialog
from utils.data_collator import DialogCollator, PlannerCollator
from model.model_color import COLOR
from model.model_dialog import DialogModel
from train.trainer_dialog import IgniteTrainer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["train", "test", "selfplay"])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_gpu', type=str2bool, default="True")
    parser.add_argument('--base_model', type=str, default="GPT2", choices=["GPT2", "DialoGPT"])

    # dataset config
    parser.add_argument('--dataset', type=str, choices=["DuRecDial2", "TGConv"])
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--plan_data', type=str, default=None, help="The planned dialog path of the testset.")
    parser.add_argument('--cache_dir', type=str, default="caches/plan/", help="The cache directory of the dataset.")
    parser.add_argument('--log_dir', type=str, default="logs/plan/", help="The log directory of the model.")
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--turn_type_size', type=int, default=16)
    parser.add_argument('--lower_case', type=str2bool, default="False")

    # training args
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--validate_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="linear", choices=['linear','noam'])
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64)

    # decoding args
    parser.add_argument('--infer_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--max_dec_len', type=int, default=100)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--diversity_penalty', type=float, default=0.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--bad_words_ids', type=list, default=None)
    parser.add_argument('--remove_invalid_values', type=str2bool, default="False")
    
    # addtional args for self-play
    parser.add_argument('--plan_log_dir', type=str, default=None)
    parser.add_argument('--infer_plan_checkpoint', type=str, default=None)
    parser.add_argument('--use_transform', type=str2bool, default="False")
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--max_transition_number', type=int, default=10)
    parser.add_argument('--use_KLD', type=str2bool, default="False")
    parser.add_argument('--infer_use_bridge', type=str2bool, default="True")
    parser.add_argument('--easy_hard_mode', type=str, default="easy", choices=["easy", "hard"])

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def print_args(args):
    logging.info("=============== Args ===============")
    for k in vars(args):
        logging.info("%s: %s" % (k, vars(args)[k]))

def set_seed(args):
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

def run_train(args):
    logging.info("=============== Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
   
    if args.base_model == "DialoGPT":
        # auto load from https://huggingface.co/microsoft/DialoGPT-small
        tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir="microsoft/DialoGPT-small", name="gpt2")
    else:
        # auto load from https://huggingface.co/gpt2
        tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir="gpt2", name="gpt2")
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    logging.info("{}: Add {} additional special tokens. The new vocab size is {}".format(type(tokenizer).__name__, num_added_tokens, args.vocab_size))
    
    # define dataset
    if args.dataset == "DuRecDial2":
        train_dataset = DuRecdialDataset4Dialog(data_path=args.train_data, data_partition="train",
            tokenizer=tokenizer, special_tokens_dict=token_id_dict,
            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
            turn_type_size=args.turn_type_size, lower_case=args.lower_case)
        dev_dataset = DuRecdialDataset4Dialog(data_path=args.dev_data, data_partition="dev",
            tokenizer=tokenizer, special_tokens_dict=token_id_dict,
            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
            turn_type_size=args.turn_type_size, lower_case=args.lower_case)
    elif args.dataset == "TGConv":
        train_dataset = TGConvDataset4Dialog(data_path=args.train_data, data_partition="train",
            tokenizer=tokenizer, special_tokens_dict=token_id_dict,
            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
            turn_type_size=args.turn_type_size, lower_case=args.lower_case)
        dev_dataset = TGConvDataset4Dialog(data_path=args.dev_data, data_partition="dev",
            tokenizer=tokenizer, special_tokens_dict=token_id_dict,
            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
            turn_type_size=args.turn_type_size, lower_case=args.lower_case)
    else:
        raise ValueError("Please specify the dataset name as `DuRecDial2` or `TGConv`.")

    # create dataloader
    collator = DialogCollator(device=device, padding_idx=args.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)

    # build model
    if args.load_checkpoint is not None:
        model_path = os.path.join(args.log_dir, "{}".format(args.load_checkpoint))
        model = torch.load(model_path)
    else:
        model = DialogModel(args=args)
    model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))
    
    # build trainer and execute model training
    trainer = IgniteTrainer(model=model, train_loader=train_loader, dev_loader=dev_loader, args=args)
    trainer.run()


def run_test(args):
    logging.info("=============== Testing ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, "{}".format(args.infer_checkpoint))
    else:
        model_path = os.path.join(args.log_dir, "best_model.bin")
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    logging.info("Model loaded from [{}]".format(model_path))

    # freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    if args.base_model == "DialoGPT":
        # auto load https://huggingface.co/microsoft/DialoGPT-small
        tokenizer, _, token_id_dict = get_tokenizer(config_dir="microsoft/DialoGPT-small", name="gpt2")
    else:
        # auto load https://huggingface.co/gpt2
        tokenizer, _, token_id_dict = get_tokenizer(config_dir="gpt2", name="gpt2")
    args.pad_token_id = token_id_dict["pad_token_id"]

    data_partition = "test"
    if args.dataset == "DuRecDial2":
        if "test_unseen" in args.test_data:
            data_partition = "test_unseen"
        elif "test_seen" in args.test_data:
            data_partition = "test_seen"

        test_dataset = DuRecdialDataset4Dialog(data_path=args.test_data, data_partition=data_partition,
            tokenizer=tokenizer, special_tokens_dict=token_id_dict, cache_dir=args.cache_dir, 
            max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size,
            is_test=True, plan_path=args.plan_data, lower_case=args.lower_case)
    elif args.dataset == "TGConv":
        test_dataset = TGConvDataset4Dialog(data_path=args.test_data, data_partition="test",
            tokenizer=tokenizer, special_tokens_dict=token_id_dict, cache_dir=args.cache_dir, 
            max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size,
            is_test=True, plan_path=args.plan_data, lower_case=args.lower_case)
    else:
        raise ValueError("Undefined dataset!")
    
    collator = DialogCollator(device=device, padding_idx=args.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator.custom_collate)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_prefix = "{}_{}.jsonl".format(str(args.base_model).lower(), data_partition)
    output_path = os.path.join(args.output_dir, output_prefix)
    with open(output_path, 'w', encoding='utf-8') as f:
        for inputs in tqdm(test_loader):
            # execute generation
            outputs = model.generate(args, inputs)
            # postprocess
            resps = convert_ids_to_tokens(outputs["response"], tokenizer)
            for resp in resps:
                resp_obj = {"response": resp}
                line = json.dumps(resp_obj, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
    logging.info("Saved output to [{}]".format(output_path))
    

def run_selfplay(args):
    logging.info("=============== Self-play Testing ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load model
    if args.infer_plan_checkpoint is not None:
        plan_model_path = os.path.join(args.plan_log_dir, args.infer_plan_checkpoint)
    else:
        plan_model_path = os.path.join(args.plan_log_dir, "planner_best_model.bin")
    
    # auto load from https://huggingface.co/facebook/bart-base
    bart_config_dir = "facebook/bart-base"
    plan_tokenizer, _, plan_token_id_dict = get_tokenizer(config_dir=bart_config_dir, name="bart")
    args.plan_vocab_size = len(plan_tokenizer)
    plan_model = COLOR.from_pretrained(bart_config_dir, args=args)
    plan_model.resize_token_embeddings(args.plan_vocab_size)
    plan_model.load_state_dict(torch.load(plan_model_path))
    plan_model.to(device)
    plan_model.eval()
    logging.info("Plan Model loaded from [{}]".format(plan_model_path))
    
    if args.infer_checkpoint is not None:
        dial_model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        dial_model_path = os.path.join(args.log_dir, "best_model.bin")
    dial_model = torch.load(dial_model_path)
    dial_model.to(device)
    dial_model.eval()
    logging.info("Dial Model loaded from [{}]".format(dial_model_path))

    if args.base_model == "DialoGPT":
        # auto load from https://huggingface.co/microsoft/DialoGPT-small
        dial_tokenizer, _, dial_token_id_dict = get_tokenizer(config_dir="microsoft/DialoGPT-small", name="gpt2")
    else:
        # auto load from https://huggingface.co/gpt2
        dial_tokenizer, _, dial_token_id_dict = get_tokenizer(config_dir="gpt2", name="gpt2")

    selfplay_plan_dataset = TGConvDataset4Planning(data_path=args.test_data, data_partition="selfplay",
                                                    tokenizer=plan_tokenizer, max_seq_len=args.max_seq_len,
                                                    turn_type_size=args.turn_type_size, 
                                                    selfplay_mode=args.easy_hard_mode)
    selfplay_plan_collator = PlannerCollator(device=device, model=plan_model, latent_dim=args.latent_dim,
                                             padding_idx=plan_token_id_dict["pad_token_id"], is_eval=True)
    selfplay_dial_dataset = TGConvDataset4Dialog(data_path=args.test_data, data_partition="selfplay",
                                                 tokenizer=dial_tokenizer, special_tokens_dict=dial_token_id_dict,
                                                 max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size,
                                                 is_test=True, lower_case=args.lower_case, selfplay_mode=args.easy_hard_mode)
    selfplay_dial_collator = DialogCollator(device=device, padding_idx=dial_token_id_dict["pad_token_id"])
    
    store_samples = []
    new_sample = {}
    pre_dialog_id = -1
    now_dialog_id = 0
    now_sample_id = 0
    reach_target = False
    all_samples = selfplay_plan_dataset.get_items()
    logging.info("Testing with {} targets.".format(args.easy_hard_mode))

    while now_sample_id < len(all_samples):
        now_dialog_id = all_samples[now_sample_id]["id"]
        if pre_dialog_id != now_dialog_id:
            print("Now dialog id: {}".format(now_dialog_id))
            if not reach_target:
                store_samples.append(new_sample)
            now_bot_turn = True
            reach_target = False
            conversation = deepcopy(all_samples[now_sample_id]["conversation"][-1:])
            new_sample = {
                "id": all_samples[now_sample_id]["id"],
                "knowledge": all_samples[now_sample_id]["knowledge"],
                "conversation": conversation,
                "topic_path": "",
                "init_conversation": deepcopy(all_samples[now_sample_id]["conversation"]),
                "generated_response_list": [],
                "plan_path_list": [],
                "target": all_samples[now_sample_id]["target"],
                "response": all_samples[now_sample_id]["response"],
            }
        elif reach_target == False:
            new_sample = {
                "id": all_samples[now_sample_id]["id"],
                "knowledge": all_samples[now_sample_id]["knowledge"],
                "conversation": deepcopy(new_sample["conversation"]),
                "topic_path": "",
                "init_conversation": deepcopy(new_sample["init_conversation"]),
                "generated_response_list": deepcopy(new_sample["generated_response_list"]),
                "plan_path_list": deepcopy(new_sample["plan_path_list"]),
                "target": all_samples[now_sample_id]["target"],
                "response": all_samples[now_sample_id]["response"],
            }
        elif reach_target == True:
            now_sample_id += 1
            continue
        if now_bot_turn:
            # plan dialogue path
            plan_inputs = selfplay_plan_dataset.parse_input_context(new_sample)
            plan_feature_inputs = PlannerInput(**plan_inputs)
            plan_tensor_inputs = selfplay_plan_collator.custom_collate([plan_feature_inputs])

            plan_outputs = plan_model.generate(plan_tensor_inputs, plan_tokenizer, args=args)
            plan_path_str = combine_tokens(plan_outputs, plan_tokenizer)[0]
            new_sample["plan_path"] = plan_path_str
            new_sample["plan_path_list"].append(plan_path_str)

            # generate dialogue response
            dial_inputs = selfplay_dial_dataset.parse_sample(new_sample)
            dial_feature_inputs = DialogInput(**dial_inputs)
            dial_tensor_inputs = selfplay_dial_collator.custom_collate([dial_feature_inputs])

            dial_outputs = dial_model.generate(args, dial_tensor_inputs)
            dial_resp = convert_ids_to_tokens(dial_outputs["response"], dial_tokenizer)[0]

            new_sample["generated_response_list"].append(dial_resp)
            new_sample["conversation"].append(dial_resp)
            if args.easy_hard_mode == "easy":
                now_bot_turn = False
            elif args.easy_hard_mode == "hard":
                now_bot_turn = True
        else:
            new_sample["conversation"].append(new_sample["response"])
            now_bot_turn = True

        if new_sample["target"].lower() in new_sample["generated_response_list"][-1].lower():
            reach_target = True
            store_samples.append(new_sample)
        else:
            reach_target = False

        pre_dialog_id = now_dialog_id

        if now_sample_id + 1 >= len(all_samples) and reach_target == False:
            store_samples.append(new_sample)
            break
        
        if args.easy_hard_mode == "easy":
            stop_flag1 = 6
            stop_flag2 = 4
        elif args.easy_hard_mode == "hard":
            stop_flag1 = 8
            stop_flag2 = 8
        if len(new_sample["generated_response_list"]) < stop_flag1 and now_dialog_id != all_samples[now_sample_id+1]["id"]:
            now_sample_id = now_sample_id
            now_bot_turn = True
        elif len(new_sample["generated_response_list"]) >= stop_flag2 and now_dialog_id == all_samples[now_sample_id+1]["id"]:
            now_sample_id = now_sample_id + 1
        else:
            now_sample_id = now_sample_id + 1
   
    store_samples = store_samples[1:]
    logging.info("Total samples: {}".format(len(store_samples)))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_path = os.path.join(args.output_dir, "selfplay_{}_target.jsonl".format(args.easy_hard_mode))

    with open(save_path, "w", encoding='utf-8') as f:
        for idx, sample in enumerate(store_samples):
            dumps_sample = {
                "id": sample["id"],
                "init_conversation": sample["init_conversation"],
                "generated_response_list": sample["generated_response_list"],
                "plan_path_list": sample["plan_path_list"],
                "target": sample["target"],
                "conversation": sample["conversation"],
            }
            f.write(json.dumps(dumps_sample) + "\n")
            f.flush()
    logging.info("Saved output to [{}]".format(save_path))


if __name__ == "__main__":
    args = parse_config()
    set_seed(args)
    
    if args.mode == "train":
        print_args(args)
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "selfplay":
        run_selfplay(args)
    else:
        exit("Please specify the \"mode\" parameter!")
