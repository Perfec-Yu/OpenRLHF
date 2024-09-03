import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, OfflinePolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer

import mlflow


class OfflinePPOTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (nn.Module): the critic model in ppo algorithm
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: Optional[nn.Module],
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        self.buffer_cpu_offload = buffer_cpu_offload

        normalize_advantages = "none" if self.args.reward_normalization.startswith("reward_only") else self.args.reward_normalization
        self.actor_loss_fn = OfflinePolicyLoss(
            eps_clip,
            not self.args.use_kl,
            normalize_advantages=normalize_advantages,
            all_reduce_op=self.strategy.all_reduce if normalize_advantages != "none" else None,
        )
        if self.critic is not None:
            self.critic_loss_fn = ValueLoss(value_clip)
        else:
            self.critic_loss_fn = None
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        # self._wandb = None
        # if self.strategy.args.use_wandb and self.strategy.is_rank_0():
        #     import wandb

        #     self._wandb = wandb
        #     if not wandb.api.api_key:
        #         wandb.login(key=strategy.args.use_wandb)
        #     wandb.init(
        #         entity=strategy.args.wandb_org,
        #         project=strategy.args.wandb_project,
        #         group=strategy.args.wandb_group,
        #         name=strategy.args.wandb_run_name,
        #         config=strategy.args.__dict__,
        #         reinit=True,
        #     )

        #     wandb.define_metric("train/global_step")
        #     wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
        #     wandb.define_metric("eval/epoch")
        #     wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        if self.args.use_mlflow and self.strategy.is_rank_0():
            mlflow.set_tracking_uri(self.args.mlflow_tracking_uri)
            mlflow.set_experiment(self.args.mlflow_experiment_name)
            mlflow.start_run(run_name=self.args.mlflow_run_name)
            mlflow.log_params(self.args.__dict__)

    def to_device(self, tensor, device):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        elif isinstance(tensor, dict):
            return {k: self.to_device(v, device) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return [self.to_device(v, device) for v in tensor]
        else:
            return tensor
    
    @torch.no_grad()
    def normalize(self, items: List[torch.Tensor], masks: Optional[List[torch.Tensor]], no_std: bool=False, rloo: bool=False) -> None:
        items_vector = torch.cat(items).float().flatten()
        if masks is None: #torch.ones_like(items_vector).sum()
            sum_and_count = torch.tensor([items_vector.sum(), torch.numel(items_vector)], device=items_vector.device)
            all_sum, all_count = self.strategy.all_reduce(sum_and_count, "sum")
            mean = all_sum / all_count
            if not no_std:
                std = ((items_vector - mean).pow(2)).sum()
        else:
            masks_vector = torch.cat(masks).flatten()
            sum_and_count = torch.tensor([items_vector.sum(), masks_vector.sum()], device=items_vector.device)
            all_sum, all_count = self.strategy.all_reduce(sum_and_count, "sum")
            mean = all_sum / all_count
            if not no_std:
                std = ((items_vector - mean).pow(2) * masks_vector).sum()
        if no_std:
            if rloo:
                return [(all_count * item - all_sum) / (all_count - 1) for item in items]
            else:
                return [(item - mean) for item in items]
        else:
            all_std = self.strategy.all_reduce(std, "sum")
            rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
            if rloo:
                return [(all_count * item - all_sum) / (all_count - 1) * rstd for item in items]
            else:
                return [(item - mean) * rstd for item in items]
    
    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        gamma = self.args.gamma
        lambd = self.args.lambd
        lastgaelam = 0
        advantages_reversed = []
        response_length = action_mask.size(1)
        
        index = torch.argmax(action_mask * torch.arange(response_length, 0, -1, device=action_mask.device), dim=1).unsqueeze(1)
        rewards_expanded = torch.zeros_like(action_mask, dtype=rewards.dtype)
        rewards_expanded.scatter_(1, index, rewards.unsqueeze(1))

        values = action_mask * values

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards_expanded[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns.detach()
    
    @torch.no_grad()
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = self.to_device(batch, "cuda")
        if self.critic is not None:
            self.critic.eval()
            values = self.critic(batch["input_ids"], num_actions=0, attention_mask=batch["attention_mask"])
            batch["values"] = (values * batch["action_mask"]).detach()
        self.initial_model.eval()
        base_action_log_probs = self.initial_model(batch["input_ids"], 0, batch["attention_mask"])
        batch["base_action_log_probs"] = (base_action_log_probs * batch["action_mask"]).detach()
        return batch
    
    @torch.no_grad()
    def prepare_rollout(self, rollout_buffer: List[Dict[str, Any]]) -> None:
        """
        This function precomputes rewards, advantages and do necessary normalization
        each batch:
        - prompt_ids_lens,
        - input_ids,
        - attention_mask,
        - rewards,
        - action_mask,
        - infos
        """
        normalized_rewards = []
        if self.args.reward_normalization == "reward_only_no_std":
            normalized_rewards = self.normalize([item["rewards"] for item in rollout_buffer], None, no_std=True, rloo=False)
        elif self.args.reward_normalization == "reward_only_rloo":
            normalized_rewards = self.normalize([item["rewards"] for item in rollout_buffer], None, no_std=False, rloo=True)
        elif self.args.reward_normalization == "reward_only_rloo_no_std":
            normalized_rewards = self.normalize([item["rewards"] for item in rollout_buffer], None, no_std=True, rloo=True)
        elif self.args.reward_normalization == "reward_only":
            normalized_rewards = self.normalize([item["rewards"] for item in rollout_buffer], None)
        
        if normalized_rewards:
            for item, reward in zip(rollout_buffer, normalized_rewards):
                item["rewards"] = reward

        if self.critic is not None:
            for batch in rollout_buffer:
                advantages, returns = self.get_advantages_and_returns(batch["values"], batch["rewards"], batch["action_mask"])
                batch["advantages"] = advantages
                batch["returns"] = returns
        else:
            for batch in rollout_buffer:
                batch["advantages"] = batch["rewards"].unsqueeze(1) * batch["action_mask"]

        return rollout_buffer

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
    ) -> None:
        """
        ORIGINAL PPO TRAINING LOOP
        ---from main---
        num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        ---------------
        num_rollouts_per_episodes: number of rollout batches per episode
        num_episodes: total number of episodes to train.
                      one episode is a complete pass through the prompts_dataloader
                      each prompt is repeated n times (see ppo datasets for details)
        rollout_batch_size: number of rollouts (generations) per update step
        max_epochs: number of repeats to train the model on each rollout batch of rollout_batch_size samples
        micro_rollout_batch_size: per rank rollout batch size. this is the batch_size of prompt_dataloader
        micro_train_batch_size: per rank train batch size. this is the batch_size of training on rollouts and also the pretrain_dataloader
        update_timesteps: number of rollout steps per update step
        steps: record the step of generating a rollout
        start_episode: the episode to start training from
        consumed_samples: the number of samples already trained on in the current episode. 
                          But is also used to resume training, thus containing history episodes when saving status.

        OFFLINE PPO:
        prompt_dataloader.batch_size = micro_train_batch_size # we don't need to generate rollouts
        # however, we need to have a buffer to store the rollouts till it reaches rollout_batch_size
        # this is useful to perform normalization
        global_step = 0
        rollout_buffer = []
        buffer_size = rollout_batch_size // micro_train_batch_size
        for episode in range(num_episodes):
            for offline_rollout_batch in prompt_dataloader:
                rollout_buffer.append(offline_rollout_batch)
                if len(rollout_buffer) == buffer_size:
                    # normalize the advantages
                    normalize(rollout_buffer)
                    # train the model
                    for batch in rollout_buffer:
                        device = torch.cuda.current_device()
                        batch.to(device)
                        status = training_step(batch, global_step)
                    # clear the buffer
                    rollout_buffer.clear()
            global_step += 1
        """


        global_steps = consumed_samples // args.rollout_batch_size + 1
        rollout_buffer = []
        start_episode = consumed_samples // len(prompts_dataloader)
        consumed_samples = consumed_samples % len(prompts_dataloader)

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        buffer_max_size = args.rollout_batch_size // args.micro_train_batch_size

        if args.eval_steps == -1:
            args.eval_steps = len(prompts_dataloader) // buffer_max_size
        if args.save_steps == -1:
            args.save_steps = float("inf")

        for episode in range(start_episode, args.num_episodes):
            if isinstance(prompts_dataloader.sampler, DistributedSampler):
                prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )
            for item in prompts_dataloader:
                item = self.prepare_batch(item)
                rollout_buffer.append(self.to_device(item, "cpu"))
                if len(rollout_buffer) == buffer_max_size:
                    rollout_buffer = self.prepare_rollout(rollout_buffer)
                    torch.cuda.empty_cache()
                    status = self.ppo_train(rollout_buffer, global_steps)
                    global_steps += 1
                    rollout_buffer.clear()
                    torch.cuda.empty_cache()
                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    pbar.set_postfix(status)

                    # logs/checkpoints
                    client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                    self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)
                pbar.update()



    def ppo_train(self, rollout_buffer: List[Dict[str, Any]], global_steps=0) -> Dict[str, float]:
        
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        
        pbar = tqdm(
            rollout_buffer,
            desc=f"Train for batch {global_steps}",
            disable=not self.strategy.is_rank_0(),
        )
        for batch in pbar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            status = self.training_step(batch, global_steps)

            # for DP
            # weighted mean for kl
            if "kl" in status:
                # status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status = self.strategy.all_reduce(status, "mean")
                # status["kl"] /= status["response_length"]

            short_status = {}

            if "policy_loss" in status:
                short_status = {
                    "pg": status["policy_loss"],
                    "rm": status["reward"],
                    "ret": status.get("return", 0),
                    "glen": status.get("response_length", 0),
                    "tlen": status.get("total_length", 0),
                    "kl": status["kl"],
                    "surr1": status["surr1_portion"],
                    "act_lr": status["actor_lr"],
                }

            if "critic_loss" in status:
                short_status["cri"] = status["critic_loss"]
                short_status["vals"] = status["values"]
                short_status["cri_lr"] = status["critic_lr"]

            if "ptx_loss" in status:
                short_status["ptx"] = status["ptx_loss"]

            status_list.append(status)
            pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, batch: Dict[str, Any], global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(batch)
        if self.critic is not None:
            status.update(self.training_step_critic(batch))
        return status

    def training_step_actor(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.actor.train()

        # actor loss
        action_log_probs, output = self.actor(
            batch["input_ids"], 0, attention_mask=batch["attention_mask"], return_output=True
        )

        # loss function
        actor_loss, estimated_kl, surr1_portion = self.actor_loss_fn(
            action_log_probs,
            batch["base_action_log_probs"],
            batch["advantages"],
            self.kl_ctl.value,
            action_mask=batch["action_mask"],
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        
        status["reward"] = batch["rewards"].mean().item()
        status["return"] = batch["returns"].mean().item() if "returns" in batch else 0
        status["kl"] = estimated_kl.item()
        status["surr1_portion"] = surr1_portion.item()
        return status

    def training_step_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.critic.train()

        # critic loss
        values, output = self.critic(
            batch["input_ids"],
            num_actions=0,
            attention_mask=batch["attention_mask"],
            return_output=True,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            batch["values"],
            batch["returns"],
            action_mask=batch["action_mask"],
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, batch["action_mask"]).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            # if self._wandb is not None and self.strategy.is_rank_0():
            #     logs = {
            #         "train/%s" % k: v
            #         for k, v in {
            #             **logs_dict,
            #             "global_step": global_step,
            #         }.items()
            #     }
            #     self._wandb.log(logs)

            # MLflow
            if self.args.use_mlflow and self.strategy.is_rank_0():
                logs = {
                    f"train/{k}": v
                    for k, v in logs_dict.items()
                }
                logs["train/global_step"] = global_step
                mlflow.log_metrics(logs, step=global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        self.strategy.save_ckpt(
            self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        )

    def end_mlflow_run(self):
        if self.strategy.args.use_mlflow and self.strategy.is_rank_0() and mlflow.active_run() is not None:
            mlflow.end_run()

    def __del__(self):
        self.end_mlflow_run()
