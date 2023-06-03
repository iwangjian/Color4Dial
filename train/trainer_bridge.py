# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import Adam


class BrownianBridgeTrainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,model, train_loader, dev_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        # parse args
        self.log_dir = args.log_dir
        self.log_steps = args.log_steps
        self.validate_steps = args.validate_steps
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.freeze_plm = args.freeze_plm

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warmup_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        for name, param in self.model.named_parameters():
            if "freeze_plm" in name or "transform_layers" in name or "feature_conversion" in name or "feature_projection" in name or "feedback_estimation" in name:
                if self.freeze_plm and "freeze_plm" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
        logging.info("Trainable parameters:" + ", ".join(trainable_params))
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        
        save_dir = os.path.join(self.log_dir, "checkpoints_bridge")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model_store_path = os.path.join(save_dir, "bridge_best_model.bin")
       
        trained_steps = 0
        valid_loss = []
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(self.train_loader):
                self.model.train()
                trained_steps += 1
                model_output = self.model.train_bridge(inputs)
                total_loss = model_output["contra_loss"]
                valid_loss.append(total_loss.item())

                if self.gradient_accumulation_steps > 0:
                    total_loss = total_loss / self.gradient_accumulation_steps
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.gradient_accumulation_steps > 0:
                    if batch_step > 0 and batch_step % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if batch_step > 0 and batch_step % self.log_steps == 0:
                    avg_valid_loss = np.mean(valid_loss)
                    log_key = "Batch Step: {}\tAvg loss: {:.3f}"
                    log_value = (batch_step, avg_valid_loss.item(),)
                    valid_loss = []
                    logging.info(log_key.format(*log_value))
                
                if trained_steps > 0 and trained_steps % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation Average Similarity: {:.3f}".format(predicts_dict["avg_similarity"]))
                    if predicts_dict["avg_similarity"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_similarity"]
                        logging.info("Epoch {} Trained Step {} -- Best Average Similarity: {:.3f}".format(
                            epoch + 1, trained_steps, self.best_metric)
                        )
                        torch.save(self.model.state_dict(), best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            
            predicts_dict = self.evaluate(loader=self.dev_loader)
            logging.info("Evaluation Average Similarity: {:.3f}".format(predicts_dict["avg_similarity"]))
            if predicts_dict["avg_similarity"] > self.best_metric:
                self.best_metric = predicts_dict["avg_similarity"]
                logging.info("Epoch {} Trained Step {} -- Best Average Similarity: {:.3f}".format(
                    epoch + 1, trained_steps, self.best_metric))
                torch.save(self.model.state_dict(), best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(save_dir, "bridge_model_epoch_%d.bin" % (epoch + 1))
            torch.save(self.model.state_dict(), model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        self.model.eval()
        similarities = []
        with torch.no_grad():
            for inputs in loader:
                user_utt_ids, user_utt_mask = inputs["user_utterance"]
                delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
                start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
                target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]
                gold_bridge_list = inputs["gold_bridge_list"][0]     # note that bzs=1
                if len(gold_bridge_list) < 1:
                    continue

                start_latent = self.model.get_time_control_embed(start_subgoal_ids, start_subgoal_mask) # [1, 768]
                target_latent = self.model.get_time_control_embed(target_subgoal_ids, target_subgoal_mask) # [1, 768]
                Z_u = self.model.get_user_utt_representation(user_utt_ids, user_utt_mask) # [1, 768]
                delta_u = self.model.get_delta_u_representation(delta_follow_ids, delta_follow_mask) # [1]
                
                simulate_bridge_points = self.model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent, T=len(gold_bridge_list)+1, Z_u=Z_u, delta_u=delta_u)
                gold_bridge_list = [start_latent] + gold_bridge_list
                assert len(simulate_bridge_points) == len(gold_bridge_list)

                for i in range(len(simulate_bridge_points)):
                    similarity = F.cosine_similarity(simulate_bridge_points[i], gold_bridge_list[i], dim=-1).item()
                    similarities.append(similarity)
        avg_similarity = np.mean(similarities)
        result = {
            "avg_similarity": avg_similarity
        }
        return result

    def evaluate_brownian_bridge(self, loader, output_path):
        self.model.eval()
        similarities = []
        with open(output_path, 'w', encoding='utf-8') as fw:
            with torch.no_grad():
                for inputs in loader:
                    similarity_list = []
                    user_utt_ids, user_utt_mask = inputs["user_utterance"]
                    delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
                    start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
                    target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]
                    gold_bridge_list = inputs["gold_bridge_list"][0]    # note that bzs=1
                    if len(gold_bridge_list) < 1:
                        continue

                    start_latent = self.model.get_time_control_embed(start_subgoal_ids, start_subgoal_mask) # [1, 768]
                    target_latent = self.model.get_time_control_embed(target_subgoal_ids, target_subgoal_mask) # [1, 768]
                    Z_u = self.model.get_user_utt_representation(user_utt_ids, user_utt_mask) # [1, 768]
                    delta_u = self.model.get_delta_u_representation(delta_follow_ids, delta_follow_mask) # [1]
                    
                    simulate_bridge_points = self.model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent, T=len(gold_bridge_list)+1, Z_u=Z_u, delta_u=delta_u)
                    gold_bridge_list = [start_latent] + gold_bridge_list
                    assert len(simulate_bridge_points) == len(gold_bridge_list)

                    for i in range(len(simulate_bridge_points)):
                        similarity = F.cosine_similarity(simulate_bridge_points[i], gold_bridge_list[i], dim=-1).item()
                        similarity_list.append(similarity)
                        similarities.append(similarity)
                    fw.write(" ".join([str(sim) for sim in similarity_list]) + "\n")
        avg_similarity = np.mean(similarities)
        return avg_similarity