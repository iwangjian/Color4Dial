# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.models.bart import BartPretrainedModel, BartModel
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Optional

from .modeling_bart import shift_tokens_right
from .modeling_bart import BartEncoder, BartDecoder
from .generation_utils import model_decode


class COLOR(BartPretrainedModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)

        # used for planner
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)

        # parse args
        self.use_transform = args.use_transform
        self.latent_dim = args.latent_dim
        self.max_transition_number = args.max_transition_number
        self.use_KLD = args.use_KLD

        # used for bridge training
        # auto load from https://huggingface.co/facebook/bart-base
        self.freeze_plm = BartModel.from_pretrained('facebook/bart-base')

        if self.use_transform:
            self.transform_layers = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
        self.feature_conversion = nn.Linear(config.d_model, config.d_model)
        self.feature_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, self.latent_dim),
        )
        self.feedback_estimation = nn.Linear(config.d_model, 1)

        self.feature_conversion.apply(self.init_linear_weights)
        self.feature_projection.apply(self.init_linear_weights)
        self.feedback_estimation.apply(self.init_linear_weights)
        
        # used for planner
        self.transition_prediction = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, self.max_transition_number)
        )
        self.tc2hidden = nn.Linear(self.latent_dim, config.d_model)

        self.transition_prediction.apply(self.init_linear_weights)
        self.tc2hidden.apply(self.init_linear_weights)

        self.eps = 1e-6
        self.C_eta = 0.0
        self.end_pin_val = 1.0

    def init_linear_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def set_plm(self, model):
        self.freeze_plm = model
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        self.freeze_plm.resize_token_embeddings(new_num_tokens)
        self.lm_head = nn.Linear(self.config.d_model, new_num_tokens, bias=False)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def _logit(self, z_0, z_t, z_T, t_0, t_t, t_T, Z_u, delta_u):
        T = t_T - t_0    # The whole time cross
        t = t_t - t_0    # The middle time cross
        assert (T >= t).all()

        alpha = (t / (T + self.eps)).view(-1, 1)   # (bsz, 1). The ratio between the middle time length and the whole time length
        delta = (z_0 + Z_u) * (1 - alpha) + z_T * (alpha) - z_t    # (bsz, dim)
        var = ((t + delta_u) * (T - t) / (T + self.eps))
        log_p =  -1/(2*var + self.eps) * (delta*delta).sum(-1) + self.C_eta
        if len(log_p.shape) > 1:     # (1, bsz)
            log_p = log_p.squeeze(0)
        log_p = log_p.unsqueeze(-1)  # should be (bsz, 1)
        return log_p 
    
    def reg_loss(self, t_, z_0, T, z_T):
        loss = 0.0
        mse_loss_f = nn.MSELoss()
        # start regularization
        start_idxs = torch.where((t_) == 0)[0]
        if start_idxs.nelement():
            vals = z_0[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        # end regularization
        end_idxs = torch.where((T) == T)[0]
        if end_idxs.nelement():
            vals = torch.abs(z_T[end_idxs, :])
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device) * self.end_pin_val)
            loss += end_reg
        return loss

    def get_constrastive_loss(self, z_0, z_t, z_T, t_0, t, T, Z_u, delta_u):
        loss = 0.0
        pos_logit = self._logit(z_0=z_0, z_t=z_t, z_T=z_T, t_0=t_0, t_t=t, t_T=T, Z_u=Z_u, delta_u=delta_u)
        count_pos = 0
        for pos_idx in range(z_T.shape[0]):
            count_pos += 1
            neg_i_logit = self._logit(z_0=z_0, z_t=z_t[pos_idx], z_T=z_T, t_0=t_0, t_t=t, t_T=T, Z_u=Z_u, delta_u=delta_u)
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[pos_idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i[0]

        if count_pos > 0:
            loss /= count_pos
        else:
            loss = 0.0
        # regularization for pinning start and end of bridge
        reg_loss = self.reg_loss(t_0, z_0, T, z_T)
        loss += reg_loss
        return loss

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length
        return avg_hidden
    
    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def get_user_utt_representation(self, input_ids, input_masks):
        output = self.freeze_plm.encoder(
            input_ids=input_ids,
            attention_mask=input_masks,
        )
        output = self.compute_masked_means(output[0], input_masks)
        if self.use_transform:
            output = self.transform_layers(output)
        z_u = self.feature_projection(self.feature_conversion(output))   # (bsz, dim)
        return z_u

    def get_delta_u_representation(self, input_ids, input_masks):
        output = self.freeze_plm.encoder(
            input_ids=input_ids,
            attention_mask=input_masks,
        )
        output = self.compute_masked_means(output[0], input_masks)
        if self.use_transform:
            output = self.transform_layers(output)
        delta_u = torch.sigmoid(self.feedback_estimation(self.feature_conversion(output)))   # (bsz, 1)
        return delta_u

    def get_time_control_embed(self, input_ids, input_masks):
        output = self.freeze_plm.encoder(
            input_ids=input_ids,
            attention_mask=input_masks,
        )
        output = self.compute_masked_means(output[0], input_masks)
        if self.use_transform:
            output = self.transform_layers(output)
        z_feat = self.feature_projection(output)   # (bsz, dim)
        return z_feat

    def get_transition_states(self, encoder_rep, input_masks):
        output = self.compute_masked_means(encoder_rep[0], input_masks)
        if self.use_transform:
            output = self.transform_layers(output)
        transition_pred = self.transition_prediction(output)
        return transition_pred

    def get_transition_number(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        trans_T = self.get_transition_states(encoder_outputs, attention_mask).argmax(dim=-1)
        return trans_T

    def train_bridge(self, inputs):
        """Train the Brownian bridge mapping model."""
        user_utt_ids, user_utt_mask = inputs["user_utterance"]
        delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
        interim_subgoal_ids, interim_subgoal_mask = inputs["interim_subgoal"]
        start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
        target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]
        interim_t = inputs["interim_t"]
        target_T = inputs["target_T"]

        Z_u = self.get_user_utt_representation(user_utt_ids, user_utt_mask)     # (bsz, dim)
        delta_u = self.get_delta_u_representation(delta_follow_ids, delta_follow_mask)   # (bsz, 1)
        Z_s0 = self.get_time_control_embed(start_subgoal_ids, start_subgoal_mask)        # (bsz, dim)
        Z_st = self.get_time_control_embed(interim_subgoal_ids, interim_subgoal_mask)    # (bsz, dim)
        Z_sT = self.get_time_control_embed(target_subgoal_ids, target_subgoal_mask)      # (bsz, dim)

        t_0 = torch.zeros_like(interim_t)
        assert t_0.shape == interim_t.shape and interim_t.shape == target_T.shape

        contrastive_loss = self.get_constrastive_loss(Z_s0, Z_st, Z_sT, t_0, interim_t, target_T, Z_u, delta_u)
     
        output_dict = {
            "contra_loss": contrastive_loss,
            "calc_size": target_T.shape[0],
        }
        return output_dict

    def simulate_brownian_bridge(self, B_0, B_T, T, Z_u, delta_u, dt=0.05, mu=0.0, sigma=1.0):
        """Simulate a Brownian bridge trajectory."""
        device = B_0.device
        if isinstance(B_0, torch.Tensor):
            B_0 = B_0.detach().cpu().numpy()
        if isinstance(B_T, torch.Tensor):
            B_T = B_T.detach().cpu().numpy()
        if isinstance(Z_u, torch.Tensor):
            Z_u = Z_u.detach().cpu().numpy()
        bridge = [B_0]
        for step in range(1, T-1):  # [1, T-1]
            dim = B_0.shape[-1]
            noise = dt*np.sqrt((step + delta_u.item())* (T-step)/T) * np.random.normal(mu, sigma, dim)
            t = step / T
            x_tp1 = (B_0 + Z_u) * (1 - t) + t * B_T + noise
            bridge += [x_tp1]
        bridge += [B_T]
        for i in range(len(bridge)):
            bridge[i] = torch.tensor(bridge[i]).to(device)
        return bridge
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bridge_embeds: Optional[torch.Tensor] = None,
        bridge_mask: Optional[torch.Tensor] = None,
        transition_number_label: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logging.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # If passed bridge_embeds and bridge_mask, then use them
        if bridge_embeds is not None and bridge_mask is not None:
            bridge_embeds = self.tc2hidden(bridge_embeds)
            bridge_embeds = nn.functional.dropout(bridge_embeds, p=self.config.dropout, training=self.training)
        else:
            bridge_embeds = None
            bridge_mask = None

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bridge_embeds=bridge_embeds,
            bridge_mask=bridge_mask,
        )
        
        # If compute KL divergence loss
        if self.use_KLD and bridge_embeds is not None and bridge_mask is not None and decoder_attention_mask is not None:
            kl_hidden_state = self.avg_pool(decoder_outputs["original_hidden_state"], decoder_attention_mask)
            kl_tc_state = self.avg_pool(bridge_embeds, bridge_mask)
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            kl_loss = kl_criterion(F.log_softmax(kl_hidden_state, dim=-1), F.softmax(kl_tc_state, dim=-1))
        else:
            kl_loss = None

        lm_logits = self.lm_head(decoder_outputs["last_hidden_state"]) + self.final_logits_bias
        masked_lm_loss, trans_loss, trans_output = None, None, None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            masked_lm_loss = loss_func(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        trans_output = self.get_transition_states(encoder_outputs, attention_mask)
        if transition_number_label is not None:
            trans_loss_func = nn.CrossEntropyLoss()
            trans_loss = trans_loss_func(trans_output, transition_number_label)

        return {
            "lm_loss": masked_lm_loss,
            "lm_logits": lm_logits,
            "trans_loss": trans_loss,
            "trans_logits": trans_output,
            "kl_loss": kl_loss,
        }

    def generate(self, inputs, tokenizer, args):
        """Model inference function."""
        input_ids, input_masks = inputs["input"]
        user_utt_ids, user_utt_mask = inputs["user_utterance"]
        delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
        start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
        target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]

        if args.infer_use_bridge:
            # use predicted number of transitions
            sample_number = self.get_transition_number(input_ids, input_masks).item()
        else:
            sample_number = 0
        start_latents = self.get_time_control_embed(start_subgoal_ids, start_subgoal_mask)
        end_latents = self.get_time_control_embed(target_subgoal_ids, target_subgoal_mask)
        Z_u = self.get_user_utt_representation(user_utt_ids, user_utt_mask)
        delta_u = self.get_delta_u_representation(delta_follow_ids, delta_follow_mask)

        if args.infer_use_bridge:
            if sample_number == 0:
                sampled_bridge_points = None
                sample_bridge_points_mask = None
            else:
                # simulate Brownian bridge trajectories
                sampled_bridge_points = self.simulate_brownian_bridge(B_0=start_latents, B_T=end_latents, T=sample_number+1, Z_u=Z_u, delta_u=delta_u)[1:]
                sample_bridge_points_mask = torch.ones(input_ids.shape[0], sample_number).to(self.device)
        else:
            sampled_bridge_points = None
            sample_bridge_points_mask = None

        output = model_decode(
            model=self,
            inputs=inputs,
            tokenizer=tokenizer,
            bridge_points=sampled_bridge_points,
            bridge_points_mask=sample_bridge_points_mask,
            sample_number=sample_number,
            args=args
        )
        
        return output