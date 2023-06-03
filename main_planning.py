# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
from utils.data_utils import get_tokenizer
from utils.dataset_durecdial import DuRecdialDataset4Bridge, DuRecdialDataset4Planning
from utils.dataset_tgconv import TGConvDataset4Bridge, TGConvDataset4Planning
from utils.data_collator import BridgeCollator, PlannerCollator
from train.trainer_bridge import BrownianBridgeTrainer
from train.trainer_planner import PlannerTrainer
from model.model_color import COLOR
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
    parser.add_argument('--mode', type=str, choices=["train_bridge", "train_planner", "infer_planner"])
    parser.add_argument('--random_seed', type=int, default=43)
    parser.add_argument('--use_gpu', type=str2bool, default="True")

    # dataset config
    parser.add_argument('--dataset', type=str, choices=["DuRecDial2", "TGConv"])
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--test_seen_data', type=str, default=None, help="Set only for DuRecDial2 dataset.")
    parser.add_argument('--test_unseen_data', type=str, default=None, help="Set only for DuRecDial2 dataset.")
    parser.add_argument('--cache_dir', type=str, default="caches/plan/", help="The cache directory of the dataset.")
    parser.add_argument('--log_dir', type=str, default="logs/plan/", help="The log directory of the model.")
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--turn_type_size', type=int, default=16)
    
    # training args
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--validate_steps', type=int, default=400)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=0)

    parser.add_argument('--train_batch_size_bridge', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--max_transition_number', type=int, default=8)
    parser.add_argument('--freeze_plm', type=str2bool, default="True")
    parser.add_argument('--eval_brownian_bridge', type=str2bool, default="True")
    parser.add_argument('--use_transform', type=str2bool, default="False")

    parser.add_argument('--load_checkpoint_bridge', type=str, default=None)
    parser.add_argument('--load_checkpoint_planner', type=str, default=None)
    parser.add_argument('--train_batch_size_planner', type=int, default=16)
    parser.add_argument('--train_use_bridge', type=str2bool, default="True")
    parser.add_argument('--use_KLD', type=str2bool, default="True")
    parser.add_argument('--use_simulated', type=str2bool, default="True")
    parser.add_argument('--trans_alpha', type=float, default=0.1)
    parser.add_argument('--gen_beta', type=float, default=1.0)
    parser.add_argument('--kl_gamma', type=float, default=1.0)

    # decoding args
    parser.add_argument('--infer_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs/plan/")
    parser.add_argument('--infer_use_bridge', type=str2bool, default="True")
    parser.add_argument('--max_dec_len', type=int, default=80)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--diversity_penalty', type=float, default=0.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--bad_words_ids', type=list, default=None)
    parser.add_argument('--remove_invalid_values', type=bool, default=False)
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

def run_train_bridge(args):
    logging.info("=============== Brownian Bridge Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # auto load from https://huggingface.co/facebook/bart-base
    bart_config_dir = "facebook/bart-base"

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=bart_config_dir, name="bart")
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    args.sep_token_id = token_id_dict["sep_token_id"]
    logging.info("{}: Add {} additional special tokens. The new vocab size is {}".format(type(tokenizer).__name__, num_added_tokens, args.vocab_size))

    # build model
    model = COLOR.from_pretrained(bart_config_dir, args=args)
    model.resize_token_embeddings(args.vocab_size)
    model.to(device)
    
    # define dataset
    if args.dataset == "DuRecDial2":
        train_dataset = DuRecdialDataset4Bridge(data_path=args.train_data, data_partition="train", tokenizer=tokenizer, 
                                                cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        dev_dataset = DuRecdialDataset4Planning(data_path=args.dev_data, data_partition="dev", tokenizer=tokenizer, 
                                                cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
    elif args.dataset == "TGConv":
        train_dataset = TGConvDataset4Bridge(data_path=args.train_data, data_partition="train", tokenizer=tokenizer, 
                                            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        dev_dataset = TGConvDataset4Planning(data_path=args.dev_data, data_partition="dev", tokenizer=tokenizer, 
                                            cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
    else:
        raise ValueError("Please specify the dataset name as `DuRecDial2` or `TGConv`.")

    # create data collator and dataloader
    bridge_collator = BridgeCollator(device=device, padding_idx=args.pad_token_id)
    planner_collator = PlannerCollator(device=device, model=model, latent_dim=args.latent_dim, 
                                       padding_idx=args.pad_token_id, is_eval=True)  # is_eval=True for evaluation
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size_bridge, 
                              shuffle=True, collate_fn=bridge_collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=1,   # batch_size=1 for evaluation
                            shuffle=False, collate_fn=planner_collator.custom_collate)
    
    # build trainer and execute model training
    bridge_trainer = BrownianBridgeTrainer(model=model, train_loader=train_loader, 
                                           dev_loader=dev_loader, args=args)
    bridge_trainer.train()

    if args.eval_brownian_bridge:
        timeshot = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if args.dataset == "DuRecDial2":
            save_dir = os.path.join(args.log_dir, "brownian_bridge_sim")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            seen_output_path = os.path.join(save_dir, "test_seen_{}.txt".format(timeshot))
            unseen_output_path = os.path.join(save_dir, "test_unseen_{}.txt".format(timeshot))
            
            test_seen_dataset = DuRecdialDataset4Planning(data_path=args.test_seen_data, data_partition="test_seen", 
                                                          tokenizer=tokenizer, cache_dir=args.cache_dir, 
                                                          max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
            test_unseen_dataset = DuRecdialDataset4Planning(data_path=args.test_unseen_data, data_partition="test_unseen", 
                                                            tokenizer=tokenizer, cache_dir=args.cache_dir, 
                                                            max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
            # batch_size=1 for evaluation
            test_seen_loader = DataLoader(test_seen_dataset, batch_size=1, shuffle=False, collate_fn=planner_collator.custom_collate)
            test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=1, shuffle=False, collate_fn=planner_collator.custom_collate)

            logging.info("Evaluate on test-seen ...")
            avg_similarity = bridge_trainer.evaluate_brownian_bridge(test_seen_loader, seen_output_path)
            logging.info("Saved to {}".format(seen_output_path))
            logging.info("Average similarity on test-seen: {}".format(avg_similarity))
            logging.info("Evaluate on test-unseen ...")
            avg_similarity = bridge_trainer.evaluate_brownian_bridge(test_unseen_loader, unseen_output_path)
            logging.info("Saved to {}".format(unseen_output_path))
            logging.info("Average similarity on test-unseen: {}".format(avg_similarity))

        elif args.dataset == "TGConv":
            save_dir = os.path.join(args.log_dir, "brownian_bridge_sim")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_path = os.path.join(save_dir, "test_{}.txt".format(timeshot))
            
            test_dataset = TGConvDataset4Planning(data_path=args.test_data, data_partition="test", 
                                                  tokenizer=tokenizer, cache_dir=args.cache_dir, 
                                                  max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
            # batch_size=1 for evaluation
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=planner_collator.custom_collate)
            
            logging.info("Evaluate on test ...")
            avg_similarity = bridge_trainer.evaluate_brownian_bridge(test_loader, output_path)
            logging.info("Saved to {}".format(output_path))
            logging.info("Average similarity: {}".format(avg_similarity))


def run_train_planner(args):    
    logging.info("=============== Planner Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # auto load from https://huggingface.co/facebook/bart-base
    bart_config_dir = "facebook/bart-base"

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=bart_config_dir, name="bart")
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    args.sep_token_id = token_id_dict["sep_token_id"]
    logging.info("{}: Add {} additional special tokens. The new vocab size is {}".format(type(tokenizer).__name__, num_added_tokens, args.vocab_size))

    # build model
    if args.load_checkpoint_planner is not None:
        # used for continue training from a checkpoint
        model_path = os.path.join(args.log_dir, "checkpoints_planner/{}".format(args.load_checkpoint_planner))
    elif args.load_checkpoint_bridge is not None:
        # use the specified bridge to initialize the planner model
        model_path = os.path.join(args.log_dir, "checkpoints_bridge/{}".format(args.load_checkpoint_bridge))
    else:
        # use the default bridge to initialize the planner model
        model_path = os.path.join(args.log_dir, "checkpoints_bridge/bridge_best_model.bin")
    logging.info("Model loaded from [{}]".format(model_path))
    
    model = COLOR.from_pretrained(bart_config_dir, args=args)
    model.resize_token_embeddings(args.vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # define dataset
    if args.dataset == "DuRecDial2":
        train_dataset = DuRecdialDataset4Planning(data_path=args.train_data, data_partition="train", tokenizer=tokenizer, 
                                                  cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        dev_dataset = DuRecdialDataset4Planning(data_path=args.dev_data, data_partition="dev", tokenizer=tokenizer, 
                                                cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
    elif args.dataset == "TGConv":
        train_dataset = TGConvDataset4Planning(data_path=args.train_data, data_partition="train", tokenizer=tokenizer, 
                                               cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        dev_dataset = TGConvDataset4Planning(data_path=args.dev_data, data_partition="dev", tokenizer=tokenizer, 
                                             cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
    else:
        raise ValueError("Please specify the dataset name as `DuRecDial2` or `TGConv`.")

    # create data collator and dataloader
    collator = PlannerCollator(device=device, model=model, latent_dim=args.latent_dim,
                               padding_idx=args.pad_token_id, is_eval=False)  # is_eval=False for training
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size_planner, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.train_batch_size_planner, shuffle=False, collate_fn=collator.custom_collate)

    trainer = PlannerTrainer(model=model, train_loader=train_loader, 
                              dev_loader=dev_loader, args=args)
    trainer.train()


def run_infer_planner(args):
    logging.info("=============== Planner Inference ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
   
    # auto load from https://huggingface.co/facebook/bart-base
    bart_config_dir = "facebook/bart-base"

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=bart_config_dir, name="bart")
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    args.sep_token_id = token_id_dict["sep_token_id"]
    logging.info("{}: Add {} additional special tokens. The new vocab size is {}".format(type(tokenizer).__name__, num_added_tokens, args.vocab_size))
    
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, "checkpoints_planner/{}".format(args.infer_checkpoint))
    else:
        model_path = os.path.join(args.log_dir, "checkpoints_planner/planner_best_model.bin")
    model = COLOR.from_pretrained(bart_config_dir, args=args)
    model.resize_token_embeddings(args.vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    logging.info("Model loaded from [{}]".format(model_path))
    
    collator = PlannerCollator(device=device, model=model, latent_dim=args.latent_dim,
                               padding_idx=args.pad_token_id, is_eval=True)  # is_eval=True for inference
    
    trainer = PlannerTrainer(model=model, train_loader=None, dev_loader=None, args=args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dataset == "DuRecDial2":
        test_seen_dataset = DuRecdialDataset4Planning(data_path=args.test_seen_data, data_partition="test_seen", tokenizer=tokenizer,
                                                      cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        test_seen_loader = DataLoader(test_seen_dataset, batch_size=1, shuffle=False, collate_fn=collator.custom_collate)
        test_unseen_dataset = DuRecdialDataset4Planning(data_path=args.test_unseen_data, data_partition="test_unseen", tokenizer=tokenizer,
                                                        cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=1, shuffle=False, collate_fn=collator.custom_collate)
        
        seen_output_path = os.path.join(args.output_dir, "plan_test_seen.jsonl")
        unseen_output_path = os.path.join(args.output_dir, "plan_test_unseen.jsonl")
        
        trainer.infer(infer_loader=test_seen_loader, tokenizer=tokenizer, output_path=seen_output_path, args=args)
        trainer.infer(infer_loader=test_unseen_loader, tokenizer=tokenizer, output_path=unseen_output_path, args=args)
    
    elif args.dataset == "TGConv":
        test_dataset = TGConvDataset4Planning(data_path=args.test_data, data_partition="test", tokenizer=tokenizer, 
                                              cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, turn_type_size=args.turn_type_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator.custom_collate)
        output_path = os.path.join(args.output_dir, "plan_test.jsonl")
        trainer.infer(infer_loader=test_loader, tokenizer=tokenizer, output_path=output_path, args=args)
    else:
        raise ValueError("Please specify the dataset name as `DuRecDial2` or `TGConv`.")


if __name__ == "__main__":
    args = parse_config()
    set_seed(args)
    
    if args.mode == "train_bridge":
        print_args(args)
        run_train_bridge(args)
    elif args.mode == "train_planner":
        print_args(args)
        run_train_planner(args)
    elif args.mode == "infer_planner":
        run_infer_planner(args)
    else:
        exit("Please specify the \"mode\" parameter!")