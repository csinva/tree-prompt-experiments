from typing import Dict, List

from abc import ABC, abstractmethod
import logging
import random
import imodels
import imodelsx.util
import imodelsx.metrics
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptStump:
    def __init__(
        self,
        args,
        prompt: str=None,
        split_strategy: str = "iprompt",
        tokenizer=None,
        max_features=10,
        assert_checks: bool = False,
        verbose: bool = True,
        model: AutoModelForCausalLM = None,
        checkpoint: str = "EleutherAI/gpt-j-6B",
        checkpoint_prompting: str = "EleutherAI/gpt-j-6B",
        verbalizer: Dict[int, str] = {0: " Negative.", 1: " Positive."},
        batch_size: int=1,
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        prompt: str
            the prompt to use (optional)
        split_strategy: str
            'iprompt' - use iprompt to split
            'manual' - use passed prompt in args.prompt
        max_features: int
            used by StumpTabular to decide how many features to save
        checkpoint: str
            the underlying model used for prediction
        model: AutoModelForCausalLM
            if this is passed, will override checkpoint
        checkpoint_prompting: str
            the model used for finding the prompt
        """
        self.args = args
        self.prompt = prompt
        assert split_strategy in ["iprompt", "cart", "linear", "manual"]
        self.split_strategy = split_strategy
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.tokenizer = tokenizer
        self.batch_size = batch_size

        # tree stuff
        self.child_left = None
        self.child_right = None
        self.verbalizer = verbalizer

        if self.verbose:
            logging.info(f"Loading model {self.checkpoint}")

    def fit(self, X_text: List[str], y, feature_names=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, "y should have more than 1 unique value"
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        input_strings = X_text
        output_strings = [self.verbalizer[int(yi)] for yi in y]

        # set self.prompt
        if not self.split_strategy == 'manual':
            # self.model = self.model.to('cpu')
            print(
                f"calling explain_dataset_iprompt with batch size {self.batch_size}"
            )
            prompts, metadata = imodelsx.explain_dataset_iprompt(
                lm=self.model,
                input_strings=input_strings,
                output_strings=output_strings,
                checkpoint=self.checkpoint,  # which language model to use
                num_learned_tokens=12,  # how long of a prompt to learn
                n_shots=1,  # number of examples in context
                n_epochs=5,  # how many epochs to search
                batch_size=self.batch_size,  # batch size for iprompt
                llm_float16=False,  # whether to load the model in float_16
                verbose=1,  # how much to print
                # sets template like ${input}${prefix}
                prefix_before_input=False,
                mask_possible_answers=True,  # only compute loss over valid output tokens
                generation_repetition_penalty=1.0,
                pop_topk_strategy="different_start_token",
                pop_criterion="acc",
                max_n_datapoints=len(input_strings),
                # on an a6000 gpu with gpt2-xl in fp16 and batch size 32,
                # 100 steps takes around 30 minutes.
                max_n_steps=200, # limit search by a fixed number of steps
            )
            # Consider just the top-32 prompts for splitting the tree.
            prompts = prompts[:32]
            
            torch.cuda.empty_cache()
            self.model = self.model.to('cuda')
            print('((sent model to cuda))')

            # save stuff
            self.prompt = prompts[0]
            print(f"Got {len(prompts)} prompts. Top prompt: `{prompts[0]}`")
            self.prompts = prompts
            self.meta = metadata

        # set value (calls self.predict, which uses self.prompt)
        self._set_value_acc_samples(X_text, y)

        return self

    def __getstate__(self):
        """Get the stump but prevent certain attributes from being pickled.

        See also https://stackoverflow.com/a/54139237/2287177
        """
        state = self.__dict__.copy()
        # Don't pickle big things
        if "model" in state:
            del state["model"]
        if "tokenizer" in state:
            del state["tokenizer"]
        if "feature_names" in state:
            del state["feature_names"]
        return state

    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        assert not self.args.prompt_source == "data_demonstrations_knn"
        preds_proba = self.predict_proba(X_text)
        return np.argmax(preds_proba, axis=1)

    def predict_with_cache(self, X_text: List[str], past_key_values) -> np.ndarray[int]:
        if self.args.prompt_source == "data_demonstrations_knn":
            #import pdb; pdb.set_trace()
            preds_proba, preds_logits = self.predict_proba_with_cache(X_text, past_key_values)
            #import pdb; pdb.set_trace()
            probs, logits = self.anchor_probs, self.anchor_logits
            labels = self.prompt[-1]
            labels = torch.LongTensor(labels) # num_anchors, K
            K = labels.shape[-1]

            cross_entropy = preds_proba @ logits.T # 1024, num_anchors
            preds_labels = cross_entropy.argsort(-1, descending=True)[:, :self.args.knn].contiguous() # 1024, knn
            preds_onehot_labels = labels.gather(0, preds_labels.view(-1, 1).expand(-1, K)) # 1024*knn, K
            preds_onehot_labels = preds_onehot_labels.view(-1, self.args.knn, K)
            preds_onehot_labels = preds_onehot_labels.sum(1).argmax(-1) # 1024
            return preds_onehot_labels.numpy()
            ###onehot_labels = preds_onehot_labels.new_zeros(preds_onehot_labels.shape[0], K).long()
            ###onehot_labels.scatter_(1, preds_onehot_labels.view(-1, 1), 1)
            ###return onehot_labels
        preds_proba = self.predict_proba_with_cache(X_text, past_key_values)
        return np.argmax(preds_proba, axis=1)

    def predict_proba(self, X_text: List[str]) -> np.ndarray[float]:
        target_strs = list(self.verbalizer.values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        if self.args.prompt_source == "data_demonstrations":
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched(
                [self.prompt + template % (x, "") for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        else:
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        # preds = np.zeros((len(X_text), len(target_token_ids)))
        # for i, x in enumerate(X_text):
        #     preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)
        #     preds[i] = self._get_logit_for_target_tokens(x + self.prompt, target_token_ids)
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def predict_proba_with_cache(self, X_text: List[str], past_key_values) -> np.ndarray[float]:
        target_strs = list(self.verbalizer.values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        if self.args.prompt_source == "data_demonstrations":
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched_with_cache(
                past_key_values,
                [template % (x, "") for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        elif self.args.prompt_source == "data_demonstrations_knn":
            #import pdb; pdb.set_trace()
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_all_tokens_batched_with_cache(
                past_key_values,
                [template % (x, "") for x in X_text],
                batch_size=self.batch_size,
            ).float()
            #import pdb; pdb.set_trace()
            return preds.softmax(dim=-1), preds.log_softmax(dim=-1)
        else:
            raise NotImplementedError
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        # preds = np.zeros((len(X_text), len(target_token_ids)))
        # for i, x in enumerate(X_text):
        #     preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)
        #     preds[i] = self._get_logit_for_target_tokens(x + self.prompt, target_token_ids)
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def calc_key_values(self, X_text: List[str]):
        # only predict based on first token of output string
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding = True

        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.args.prompt_source == "data_demonstrations" or self.args.prompt_source == 'data_demonstrations_knn':
            p = self.prompt
            if self.args.prompt_source == 'data_demonstrations_knn':
                p = self.prompt[0]
            template = self.args.template_data_demonstrations
            if self.args.dataset_name.startswith("knnp__"):
                max_len_verb = max(len(self.tokenizer.encode(v)) for v in self.verbalizer.values())
                max_len_input = max_len_verb + max(len(self.tokenizer.encode(s)) for s in X_text) + 1
            else:
                max_len_input = -1
                for v in self.verbalizer.values():
                    max_len_input = max(max_len_input, max([len(self.tokenizer.encode(template % (s, v))) for s in X_text[:1000]]))
            max_total_len = self.model.config.n_positions
            max_len_prompt = max_total_len - max_len_input
            if True:#'dbpedia' in self.args.dataset_name or max_len_prompt < 0: # dbpedia
                print ('max len prompt less than 0, truncating to the left')
                #import pdb; pdb.set_trace()
                max_len_input = -1
                for v in self.verbalizer.values():
                    a = [len(self.tokenizer.encode(template % (s, v))) for s in X_text[:1000]]
                    max_len_input = max(max_len_input, np.percentile(a, 95))
            max_len_input = int(math.ceil(max_len_input))
            max_len_prompt = max_total_len - max_len_input
            self.max_len_input = max_len_input
            print (f'max_len_prompt: {max_len_prompt}, max_len_input: {max_len_input}')
            #import pdb; pdb.set_trace()
            assert max_len_prompt > 0
            inputs = self.tokenizer(
                [p,],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len_prompt,
                return_attention_mask=True,
            ).to(self.model.device)

            # shape is (batch_size, seq_len, vocab_size)
            #logits = self.model(**inputs)["logits"].detach()
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs['past_key_values']
        else:
            raise NotImplementedError
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        # preds = np.zeros((len(X_text), len(target_token_ids)))
        # for i, x in enumerate(X_text):
        #     preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)
        #     preds[i] = self._get_logit_for_target_tokens(x + self.prompt, target_token_ids)
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def _get_logit_for_target_tokens_batched(
        self, prompts: List[str], target_token_ids: List[int], batch_size: int = 64
    ) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc='getting dataset predictions for top prompt', colour="red"
        )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start:batch_end]
            self.tokenizer.padding = True
            self.tokenizer.truncation_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            ).to(self.model.device)

            # shape is (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                logits = self.model(**inputs)["logits"]

            token_output_positions = inputs["attention_mask"].sum(axis=1)
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position, token_output_id].item()
                        for token_output_id in target_token_ids
                    ]
                )

    def _get_logit_for_target_tokens_batched_with_cache(
        self, past_key_values, prompts: List[str], target_token_ids: List[int], batch_size: int = 64
    ) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc="getting predictions", colour="red"
        )

        past_key_values_new = []
        for i in range(len(past_key_values)):
            past_key_values_new.append( [past_key_values[i][0].expand(batch_size, -1, -1, -1), past_key_values[i][1].expand(batch_size, -1, -1, -1)] )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start:batch_end]
            if len(prompts_batch) != past_key_values_new[0][0].shape[0]:
                for i in range(len(past_key_values)):
                    past_key_values_new[i] = [past_key_values[i][0].expand(len(prompts_batch), -1, -1, -1), past_key_values[i][1].expand(len(prompts_batch), -1, -1, -1)] 
            self.tokenizer.padding = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            ).to(self.model.device)

            attention_mask = inputs['attention_mask']
            attention_mask = torch.cat((attention_mask.new_zeros(len(prompts_batch), past_key_values[0][0].shape[-2]).fill_(1), attention_mask), dim=-1)
            inputs['attention_mask'] = attention_mask

            # shape is (batch_size, seq_len, vocab_size)
            # print(">>", past_key_values[0][0].shape)
            # print({ k: v.shape for k,v in inputs.items() })
            with torch.no_grad():
                outputs = self.model(**inputs, past_key_values=past_key_values_new)
            logits = outputs["logits"]
            token_output_positions = inputs["attention_mask"].sum(axis=1) - past_key_values[0][0].shape[-2]
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position, token_output_id].item()
                        for token_output_id in target_token_ids
                    ]
                )

    def _get_logit_for_all_tokens_batched_with_cache(
        self, past_key_values, prompts: List[str], batch_size: int = 64
    ):
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc="getting predictions", colour="red"
        )

        past_key_values_new = []
        for i in range(len(past_key_values)):
            past_key_values_new.append( [past_key_values[i][0].expand(batch_size, -1, -1, -1), past_key_values[i][1].expand(batch_size, -1, -1, -1)] )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return torch.stack(logit_targets_list, dim=0)

            prompts_batch = prompts[batch_start:batch_end]
            if len(prompts_batch) != past_key_values_new[0][0].shape[0]:
                for i in range(len(past_key_values)):
                    past_key_values_new[i] = [past_key_values[i][0].expand(len(prompts_batch), -1, -1, -1), past_key_values[i][1].expand(len(prompts_batch), -1, -1, -1)] 
            self.tokenizer.padding = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            assert self.tokenizer.truncation_side == 'left'
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len_input,
                return_attention_mask=True,
            ).to(self.model.device)

            attention_mask = inputs['attention_mask']
            attention_mask = torch.cat((attention_mask.new_zeros(len(prompts_batch), past_key_values[0][0].shape[-2]).fill_(1), attention_mask), dim=-1)
            inputs['attention_mask'] = attention_mask

            # shape is (batch_size, seq_len, vocab_size)
            # print(">>", past_key_values[0][0].shape)
            # print({ k: v.shape for k,v in inputs.items() })
            with torch.no_grad():
                outputs = self.model(**inputs, past_key_values=past_key_values_new)
            logits = outputs["logits"]
            token_output_positions = inputs["attention_mask"].sum(axis=1) - past_key_values[0][0].shape[-2]
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(logits[i, token_output_position, :].cpu())
    # def _get_logit_for_target_tokens(self, prompt: str, target_token_ids: List[int]) -> np.ndarray[float]:
    #     """Get logits for each target token
    #     This can fail when token_output_ids represents multiple tokens
    #     So things get mapped to the same id representing "unknown"
    #     """
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #     logits = self.model(**inputs)['logits'].detach()  # shape is (batch_size, seq_len, vocab_size)
    #     logit_targets = [logits[0, -1, token_output_id].item() for token_output_id in target_token_ids]
    #     return np.array(logit_targets)

    def _get_first_token_id(self, prompt: str) -> str:
        """Get first token id in prompt"""
        return self.tokenizer(prompt)["input_ids"][0]

    def __str__(self):
        return f"PromptStump(val={self.value_mean:0.2f} n={np.sum(self.n_samples)} prompt={self.prompt})"

    def get_str_simple(self):
        return self.prompt

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump."""
        idxs_right = self.predict(X_text).astype(bool)
        n_right = idxs_right.sum()
        if n_right == 0 or n_right == y.size:
            self.failed_to_split = True
            return
        else:
            self.failed_to_split = False
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        self.n_samples = [y.size - idxs_right.sum(), idxs_right.sum()]
        self.acc = accuracy_score(y, 1 * idxs_right)
