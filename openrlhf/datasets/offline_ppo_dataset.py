from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import zero_pad_sequences


def preprocess_data_gemma(data, apply_chat_template, input_key='prompt', output_key='responses', reward_key='reward'):
    eot_token = '<end_of_turn>'
    prompt = data[input_key]
    if len(prompt) > 0 and prompt[0]['role'] == 'system':
        if len(prompt) == 1:
            prompt = "<bos><start_of_turn>system\n" + prompt[0]["content"] + "<end_of_turn>\n<start_of_turn>model\n"
        else:
            if prompt[1]["role"] == "user":
                prompt = "<bos><start_of_turn>system\n" + prompt[0]["content"] + "<end_of_turn>\n" + \
                    apply_chat_template(prompt[1:], tokenize=False, add_generation_prompt=True)[5:] # ignore '<bos>' token
            else:
                prompt[0]["role"] = "user"
                prompt = apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt = apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    responses = [t + eot_token for t in data[output_key]]
    rewards = data[reward_key]
    return prompt, responses, rewards


def preprocess_data_llama(data, apply_chat_template, input_key='prompt', output_key='responses', reward_key='reward'):
    eot_token = '<|eot_id|>'
    prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
    responses = [t + eot_token for t in data[output_key]]
    rewards = data[reward_key]
    return prompt, responses, rewards


def preprocess_data(data, input_template=None, input_key="prompt", output_key='response', reward_key='reward', apply_chat_template=None, model_class: Literal['llama', 'gemma'] = 'llama'):
    if input_template:
        apply_chat_template = lambda x, *args, **kwargs: input_template.format(x)

    if model_class == 'llama':
        return preprocess_data_llama(data, apply_chat_template, input_key, output_key, reward_key)
    elif model_class == 'gemma':
        return preprocess_data_gemma(data, apply_chat_template, input_key, output_key, reward_key)
    else:
        raise ValueError(f"Invalid model class: {model_class}")
    

class OfflinePPODataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        num_processors=8,  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.reward_key = getattr(self.strategy.args, "reward_key", None)
        self.model_class = getattr(self.strategy.args, "model_class", "llama")
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 10)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.reward = processed_dataset["reward"]

    def process_data(self, data):
        prompt, responses, rewards = preprocess_data(
            data,
            self.input_template,
            self.input_key,
            self.output_key,
            self.reward_key,
            apply_chat_template=self.apply_chat_template,
            model_class=self.model_class
        )
        if prompt is None or len(responses) != len(rewards) or len(responses) == 0:
            return {"prompt": None, "response": None, "reward": None, "prompt_ids_len": None}
        
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        response_tokens = [
            self.tokenizer(
                response,
                max_length=self.max_length,
                padding=False,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            for response in responses
        ]
        response_ids_lens = [response_token["attention_mask"].int().sum().item() for response_token in response_tokens]

        responses = [t for t, l in zip(responses, response_ids_lens) if l + prompt_ids_len < self.max_length]
        rewards = [r for r, l in zip(rewards, response_ids_lens) if l + prompt_ids_len < self.max_length]

        # filter the sample whose length is greater than max_length (2 for answer length)
        if not prompt or not responses or not rewards or len(responses) == 0:
            prompt = None

        # boostrap or downsample responses to self.n_samples_per_prompt, also adjust rewards accordingly
        if len(responses) > self.n_samples_per_prompt:
            sampled_indices = np.random.choice(len(responses), self.n_samples_per_prompt, replace=False)
            responses = [responses[i] for i in sampled_indices]
            rewards = [rewards[i] for i in sampled_indices]
        elif len(responses) < self.n_samples_per_prompt and len(responses) > 0:
            sampled_indices = np.random.choice(len(responses), self.n_samples_per_prompt, replace=True)
            responses = [responses[i] for i in sampled_indices]
            rewards = [rewards[i] for i in sampled_indices]

        return {"prompt": prompt, "response": responses, "reward": rewards, "prompt_ids_len": prompt_ids_len}

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx // self.n_samples_per_prompt]
        prompt = self.prompts[idx // self.n_samples_per_prompt]
        response = self.responses[idx // self.n_samples_per_prompt][idx % self.n_samples_per_prompt]
        reward = self.reward[idx // self.n_samples_per_prompt][idx % self.n_samples_per_prompt]

        prompt_token = self.tokenizer.encode(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        response_token = self.tokenizer.encode(
            response,
            padding=False,
            truncation=False,
            add_special_tokens=False
        )
        action_length = len(response_token)

        sequence = torch.tensor(prompt_token + response_token, dtype=torch.long)
        info = {"input": prompt, "output": response, "input_length": prompt_ids_len + action_length}

        return prompt_ids_len, sequence, action_length, reward, info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        rewards = []
        action_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, action_length, reward, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(torch.tensor([1] * len(input_id), dtype=torch.long))
            rewards.append(reward)
            action_masks.append(torch.tensor([0] * (prompt_ids_len-1) + [1] * action_length, dtype=torch.long))
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        action_masks = zero_pad_sequences(action_masks, "right")
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return {
            "prompt_ids_lens": prompt_ids_lens,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "rewards": rewards,
            "action_mask": action_masks,
            "infos": infos,
        }

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        rewards = []
        action_masks = []
        infos = {"input_length": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.ones_like(input_id.flatten()) * index)
            prompt_ids_lens.append(prompt_ids_len)
            rewards.append(info["reward"])
            action_masks.append(torch.tensor([0] * prompt_ids_len + [1] * info["action_length"], dtype=torch.long))
            infos["input_length"].append(info["input_length"])
            index += 1

        # Concatenate all tensors into a single row
        # https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/llama/modeling_llama.py#L1028
        packed_input_ids.append(torch.tensor([self.tokenizer.pad_token_id]))
        packed_attention_masks.append(torch.tensor([0]))

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        action_masks = zero_pad_sequences(action_masks, "right")

        return {
            "prompt_ids_lens": prompt_ids_lens,
            "packed_input_ids": packed_input_ids,
            "packed_attention_mask": packed_attention_masks,
            "rewards": rewards,
            "action_mask": action_masks,
            "infos": infos,
        }