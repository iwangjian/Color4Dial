# -*- coding: utf-8 -*-
import logging
import os
import json
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils.data_utils import combine_tokens, get_eval_output

        
class PlannerTrainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self, model, train_loader, dev_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        # parse args
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.log_steps = args.log_steps
        self.validate_steps = args.validate_steps
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        
        self.alpha = args.trans_alpha
        self.beta = args.gen_beta
        self.gamma = args.kl_gamma
        self.train_use_bridge = args.train_use_bridge
        self.use_simulated = args.use_simulated
        self.use_KLD = args.use_KLD

        if train_loader is not None:
            total_steps = len(train_loader) * self.num_epochs
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.warmup_ratio * total_steps, num_training_steps=total_steps)
            self.best_metric = 0.0
            self.store_best_mode = "LM acc"  # or "transition acc"

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        for name, param in self.model.named_parameters():
            if "freeze_plm" in name or "feature_conversion" in name or "feedback_estimation" in name or "feature_projection" in name or "transform_layers" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))

        save_dir = os.path.join(self.log_dir, "checkpoints_planner")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model_store_path = os.path.join(save_dir, "planner_best_model.bin")
        
        train_steps = 0
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(self.train_loader):
                self.model.train()
                train_steps += 1

                input_ids, input_masks = inputs["input"]
                decoder_input_all_ids, decoder_input_all_masks = inputs["decoder_input_all"]
                labels, _ = inputs["label"]
                transition_number_label = inputs["transition_number"]
                simulate_bridge_embed, simulate_bridge_mask = inputs["simulate_bridge_embed"]
                gold_bridge_embed, gold_bridge_mask = inputs["gold_bridge_embed"]
                    
                if self.use_simulated:
                    model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                              decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                              bridge_embeds=simulate_bridge_embed, bridge_mask=simulate_bridge_mask, 
                                              transition_number_label=transition_number_label)
                else:
                    model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                              decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks, labels=labels, 
                                              bridge_embeds=gold_bridge_embed, bridge_mask=gold_bridge_mask, 
                                              transition_number_label=transition_number_label)

                lm_loss = model_output["lm_loss"]
                if self.train_use_bridge:
                    trans_loss = model_output["trans_loss"]
                    if self.use_KLD:
                        kl_loss = model_output["kl_loss"]
                        total_loss = self.alpha * trans_loss + self.beta * lm_loss + self.gamma * kl_loss
                    else:
                        total_loss = self.alpha * trans_loss + self.beta * lm_loss
                else:
                    total_loss = lm_loss

                if self.gradient_accumulation_steps > 0:
                    total_loss = total_loss / self.gradient_accumulation_steps
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.gradient_accumulation_steps > 0:
                    if batch_step > 0 and batch_step % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if train_steps > 0 and train_steps % self.log_steps == 0:
                    if self.train_use_bridge:
                        if self.use_KLD:
                            log_key = "Train Step: {}\ttotal_loss: {:.3f} lm_loss: {:.3f} trans_loss: {:.3f} kl_loss: {:.3f}"
                            log_value = (train_steps, total_loss.item(), lm_loss.item(), trans_loss.item(), kl_loss.item())
                        else:
                            log_key = "Train Step: {}\ttotal_loss: {:.3f} lm_loss: {:.3f} trans_loss: {:.3f}"
                            log_value = (train_steps, total_loss.item(), lm_loss.item(), trans_loss.item())
                    else:
                        log_key = "Train Step: {}\ttotal_loss: {:.3f} lm_loss: {:.3f}"
                        log_value = (train_steps, total_loss.item(), lm_loss.item())
                    logging.info(log_key.format(*log_value))
                
                if train_steps > 0 and train_steps % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation LM Acc: {:.3f} LM loss: {:.3f} Trans Acc: {:.3f} Trans loss: {:.3f} Total loss: {:.3f}".format(
                        predicts_dict["avg_lm_acc"], predicts_dict["avg_lm_loss"], 
                        predicts_dict["avg_trans_acc"], predicts_dict["avg_trans_loss"], 
                        predicts_dict["avg_total_loss"]))

                    if self.store_best_mode == "LM acc":
                        if predicts_dict["avg_lm_acc"] > self.best_metric:
                            self.best_metric = predicts_dict["avg_lm_acc"]
                            logging.info("Epoch {} Train Step {} -- Best LM Acc: {:.3f} -- Best LM loss: {:.3f} -- Best Trans Acc: {:.3f} -- Best Trans loss: {:.3f} -- Best Total loss: {:.3f}".format(
                                epoch + 1, train_steps, predicts_dict["avg_lm_acc"], predicts_dict["avg_lm_loss"],
                                  predicts_dict["avg_trans_acc"], predicts_dict["avg_trans_loss"], predicts_dict["avg_total_loss"]))
                            torch.save(self.model.state_dict(), best_model_store_path)
                            logging.info("Best model saved to {}".format(best_model_store_path))
                    elif self.store_best_mode == "transition acc":
                        if predicts_dict["avg_trans_acc"] > self.best_metric:
                            self.best_metric = predicts_dict["avg_trans_acc"]
                            logging.info("Epoch {} Train Step {} -- Best LM Acc: {:.3f} -- Best LM loss: {:.3f} -- Best Trans Acc: {:.3f} -- Best Trans loss: {:.3f} -- Best Total loss: {:.3f}".format(
                                epoch + 1, train_steps, predicts_dict["avg_lm_acc"], predicts_dict["avg_lm_loss"], 
                                predicts_dict["avg_trans_acc"], predicts_dict["avg_trans_loss"], predicts_dict["avg_total_loss"]))
                            torch.save(self.model.state_dict(), best_model_store_path)
                            logging.info("Best model saved to {}".format(best_model_store_path))
                    else:
                        raise Exception("Unknown store best mode!")

            predicts_dict = self.evaluate(loader=self.dev_loader)
            
            if self.store_best_mode == "LM acc":
                if predicts_dict["avg_lm_acc"] > self.best_metric:
                    self.best_metric = predicts_dict["avg_lm_acc"]
                    logging.info("Epoch {} Train Step {} -- Best LM Acc: {:.3f} -- Best LM loss: {:.3f} -- Best Trans Acc: {:.3f} -- Best Trans loss: {:.3f} -- Best Total loss: {:.3f}".format(
                        epoch + 1, train_steps, predicts_dict["avg_lm_acc"], predicts_dict["avg_lm_loss"], 
                        predicts_dict["avg_trans_acc"], predicts_dict["avg_trans_loss"], predicts_dict["avg_total_loss"]))
                    torch.save(self.model.state_dict(), best_model_store_path)
                    logging.info("Best model saved to {}".format(best_model_store_path))
            elif self.store_best_mode == "transition acc":
                if predicts_dict["avg_trans_acc"] > self.best_metric:
                    self.best_metric = predicts_dict["avg_trans_acc"]
                    logging.info("Epoch {} Train Step {} -- Best LM Acc: {:.3f} -- Best LM loss: {:.3f} -- Best Trans Acc: {:.3f} -- Best Trans loss: {:.3f} -- Best Total loss: {:.3f}".format(
                        epoch + 1, train_steps, predicts_dict["avg_lm_acc"], predicts_dict["avg_lm_loss"],
                        predicts_dict["avg_trans_acc"], predicts_dict["avg_trans_loss"], predicts_dict["avg_total_loss"]))
                    torch.save(self.model.state_dict(), best_model_store_path)
                    logging.info("Best model saved to {}".format(best_model_store_path))
            else:
                raise Exception("Unknown store best mode!")
            
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(save_dir, "plan_model_epoch_%d.bin" % (epoch + 1))
            torch.save(self.model.state_dict(), model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def compute_acc(self, lm_logits, seq_masks, seq_labels):
        pred = torch.softmax(lm_logits, -1)
        _, pred_y = pred.max(-1)
        hit_tokens = (torch.eq(pred_y, seq_labels).float()*seq_masks).sum().item()
        num_tokens = seq_masks.float().sum().item()
        acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
        return acc

    def compute_trans_acc(self, transition_logits, transition_labels):
        pred = torch.softmax(transition_logits, -1)
        _, pred_y = pred.max(-1)
        hit_tokens = (torch.eq(pred_y, transition_labels).float()).sum().item()
        num_tokens = transition_labels.size(0)
        acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
        return acc

    def evaluate(self, loader):
        self.model.eval()
        lm_accs, lm_losss, trans_accs, trans_losss, total_losss = [], [], [], [], []
        with torch.no_grad():
            for inputs in loader:
                input_ids, input_masks = inputs["input"]
                decoder_input_all_ids, decoder_input_all_masks = inputs["decoder_input_all"]
                labels, labels_mask = inputs["label"]
                transition_number_label = inputs["transition_number"]
                simulate_bridge_embed, simulate_bridge_mask = inputs["simulate_bridge_embed"]
                gold_bridge_embed, gold_bridge_mask = inputs["gold_bridge_embed"]

                if self.use_simulated:
                    model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                              decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                              bridge_embeds=simulate_bridge_embed, bridge_mask=simulate_bridge_mask, 
                                              transition_number_label=transition_number_label)
                else:
                    model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                              decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks, labels=labels, 
                                              bridge_embeds=gold_bridge_embed, bridge_mask=gold_bridge_mask, 
                                              transition_number_label=transition_number_label)

                lm_logits = model_output["lm_logits"]
                lm_loss = model_output["lm_loss"]
                if self.train_use_bridge:
                    trans_logits = model_output["trans_logits"]
                    trans_loss = model_output["trans_loss"]
                    trans_acc = self.compute_trans_acc(trans_logits, transition_number_label)
                    if self.use_KLD:
                        kl_loss = model_output["kl_loss"]
                        total_loss = self.alpha * trans_loss + self.beta * lm_loss + self.gamma * kl_loss
                    else:
                        total_loss = self.alpha * trans_loss + self.beta * lm_loss
                else:
                    trans_loss = 0.0
                    trans_acc = 0.0
                    kl_loss = 0.0
                    total_loss = lm_loss
                lm_acc = self.compute_acc(lm_logits, labels_mask, labels)
                lm_accs.append(lm_acc)
                trans_accs.append(trans_acc)

                if self.gradient_accumulation_steps > 0:
                    lm_losss.append(float(lm_loss) / self.gradient_accumulation_steps)
                    trans_losss.append(float(trans_loss) / self.gradient_accumulation_steps)
                    total_losss.append(float(total_loss) / self.gradient_accumulation_steps)
                else:
                    lm_losss.append(float(lm_loss))
                    trans_losss.append(float(trans_loss))
                    total_losss.append(float(total_loss))
        avg_lm_acc = np.mean(lm_accs)
        avg_lm_loss = np.mean(lm_losss)
        avg_trans_acc = np.mean(trans_accs)
        avg_trans_loss = np.mean(trans_losss)
        avg_total_loss = np.mean(total_losss)
        
        return_dict = {
            "avg_lm_acc": avg_lm_acc,
            "avg_lm_loss": avg_lm_loss,
            "avg_trans_acc": avg_trans_acc,
            "avg_trans_loss": avg_trans_loss,
            "avg_total_loss": avg_total_loss,
        }
        return return_dict

    def infer(self, infer_loader, tokenizer, output_path, args):
        self.model.eval()
         
        trans_accs = []
        with open(output_path, "w", encoding="utf-8") as fw:
            with torch.no_grad():
                for inputs in tqdm(infer_loader):
                    input_ids, input_masks = inputs["input"]
                    decoder_input_all_ids, decoder_input_all_masks = inputs["decoder_input_all"]
                    transition_number_label = inputs["transition_number"]
                    simulate_bridge_embed, simulate_bridge_mask = inputs["simulate_bridge_embed"]
                    gold_bridge_embed, gold_bridge_mask = inputs["gold_bridge_embed"]

                    if self.use_simulated:
                        model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                                decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks,
                                                bridge_embeds=simulate_bridge_embed, bridge_mask=simulate_bridge_mask, 
                                                transition_number_label=transition_number_label)
                    else:
                        model_output = self.model(input_ids=input_ids, attention_mask=input_masks, 
                                                decoder_input_ids=decoder_input_all_ids, decoder_attention_mask=decoder_input_all_masks, 
                                                bridge_embeds=gold_bridge_embed, bridge_mask=gold_bridge_mask, 
                                                transition_number_label=transition_number_label)

                    if self.train_use_bridge:
                        trans_logits = model_output["trans_logits"]
                        trans_acc = self.compute_trans_acc(trans_logits, transition_number_label)
                    else:
                        trans_acc = 0.0
                    trans_accs.append(trans_acc)

                    generated_output = self.model.generate(inputs, tokenizer, args=args)
                    sentences = combine_tokens(generated_output, tokenizer)
                        
                    if self.dataset == "DuRecDial2":
                        for sentence in sentences:
                            action, topic = get_eval_output(sentence)
                            plan = {
                                "action": action,
                                "topic": topic,
                                "plan_path": sentence
                            }
                            line = json.dumps(plan, ensure_ascii=False)
                            fw.write(line + "\n")
                            fw.flush()
                    else:
                        for sentence in sentences:
                            topic = get_eval_output(sentence, topic_only=True)
                            plan = {
                                "topic": topic,
                                "plan_path": sentence
                            }
                            line = json.dumps(plan, ensure_ascii=False)
                            fw.write(line + "\n")
                            fw.flush()
        avg_trans_acc = np.mean(trans_accs)
        logging.info("Transition acc: {:.3f}".format(avg_trans_acc))
        logging.info("Saved output to {}".format(output_path))
