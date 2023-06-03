# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
from transformers.optimization import AdamW

from torch.optim.lr_scheduler import LambdaLR
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler


class IgniteTrainer(object):
    
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            args
        ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        # parse args
        self.log_dir = args.log_dir
        self.validate_steps = args.validate_steps
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': self.lr}], lr=self.lr, correct_bias=True)
            
        if args.scheduler == "noam":
            # noam decrease the learning rate
            noam_lambda = lambda step: (
                    args.hidden_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
            noam_scheduler = LambdaLR(self.optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
            self.scheduler = LRScheduler(noam_scheduler)
        else:
            # linear decrease the learning rate
            self.scheduler = PiecewiseLinear(self.optimizer, "lr", [(0, self.lr), (self.num_epochs * len(train_loader), 0.0)])

    def update(self, engine, batch):
        self.model.train()
        model_output = self.model(batch)
        loss = model_output["loss"]
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if engine.state.iteration % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return (loss.item(), self.optimizer.param_groups[0]['lr'])

    def inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(batch)
            lm_loss = model_output["lm_loss"]
            acc = model_output["acc"]
            return (lm_loss.item(), acc)

    def run(self):
        trainer = Engine(self.update)
        evaluator = Engine(self.inference)

        # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(self.dev_loader))
        if self.num_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(self.dev_loader))

        # Evaluation during training
        @trainer.on(Events.ITERATION_STARTED)
        def log_iterations(engine):
            if engine.state.iteration % self.validate_steps == 0:
                evaluator.run(self.dev_loader)
        
        trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler)

        # Prepare metrics
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss") # update() -> loss
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")   # update() -> lr
        metrics = {"avg_nll": RunningAverage(output_transform=lambda x: x[0])}  # inference() -> lm_loss
        metrics["avg_ppl"] = MetricsLambda(np.math.exp, metrics["avg_nll"])
        metrics["avg_acc"] = RunningAverage(output_transform=lambda x: x[1])    # inference() -> acc
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        # On the main process: add progress bar, tensorboard, checkpoints
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: {}".format(evaluator.state.metrics)))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        tb_logger = TensorboardLogger(log_dir=self.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                        event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(self.optimizer),
                        event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                        another_engine=trainer),
                        event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoints', save_interval=1, n_saved=3)
        
        # Save model after evaluation
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'model': self.model})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'model': self.model})

        # Run training
        trainer.run(self.train_loader, max_epochs=self.num_epochs)

        # On the main process: save the last checkpoint
        best_model_store_path = os.path.join(tb_logger.writer.logdir, "best_model.bin")
        torch.save(self.model, best_model_store_path)
        logging.info("Saved to [%s]" % best_model_store_path)

        tb_logger.close()