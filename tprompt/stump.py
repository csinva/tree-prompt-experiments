from typing import List

from abc import ABC, abstractmethod
import logging

import imodels
import imodelsx.util
import imodelsx.metrics
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class Stump(ABC):
    def __init__(
        self,
        args,
        split_strategy: str = 'iprompt',
        tokenizer=None,
        max_features=10,
        assert_checks: bool = False,
        verbose: bool = True,
        model: AutoModelForCausalLM = None,
        checkpoint: str = 'EleutherAI/gpt-j-6B',
        checkpoint_prompting: str = 'EleutherAI/gpt-j-6B',
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        split_strategy: str
            'iprompt' - use iprompt to split
            'manual' - use passed prompt in args.prompt
            'cart' - use cart to split
            'linear' - use linear to split
        max_features: int
            used by StumpTabular to decide how many features to save
        checkpoint: str
            the underlying model used for prediction
        model: AutoModelForCausalLM
            if this is passed, will override checkpoint
        checkpoint_prompting: str
            the model used for finding the prompt
        """
        print("[0] Stump::init()")
        self.args = args
        assert split_strategy in ['iprompt', 'cart', 'linear', 'manual']
        self.split_strategy = split_strategy
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.tokenizer = tokenizer

        # tree stuff
        self.child_left = None
        self.child_right = None
        print("[1] Stump::init()")

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

    @abstractmethod
    def fit(self, X_text: List[str], y: List[int], feature_names=None, X=None):
        return self

    @abstractmethod
    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        return

    def _set_value_acc_samples(self, model, X_text, y):
        """Set value and accuracy of stump.
        """
        idxs_right = self.predict(model, X_text).astype(bool)
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


class PromptStump(Stump):

    def __init__(self, *args, **kwargs):
        super(PromptStump, self).__init__(*args, **kwargs)
        if self.verbose:
            logging.info(f'Loading PromptStump with checkpoint {self.checkpoint}')
        
    def get_llm_prompt_suffix(self) -> str:
        from tprompt.prompts import get_prompts
        base_prompts = get_prompts(self.args, None, None, None, verbalizer_num=self.args.verbalizer_num, seed=1)
        return f"Prompts:\n- {base_prompts[0]}\n- {base_prompts[1]}\n- {base_prompts[2]}\n-"

    def fit(self, model, X_text: List[str], y, feature_names=None, X=None):
        print("PromptStump fit()")
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        # X, y, _ = imodels.util.arguments.check_fit_arguments(
        #     self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        input_strings = X_text
        verbalizer_dict = self._get_verbalizer()
        output_strings = [verbalizer_dict[int(yi)] for yi in y]

        # get prompt
        if self.split_strategy == 'manual':
            self.prompt = self.args.prompt
        else:
            # TODO: Load
            # llm_prompt = 'Prompt:'
            llm_prompt_start = "What question would you give to a language model to distinguish between Group 1 and Group 2? Please read the text (a movie review) and respond with a distinguishing question that applies to all datapoints and is different than previosuly written. Don't make any references to groups like Group 1 or Group 2. The answer to the question be Yes or No. Try to be more specific as specific as possible in your question (just asking 'is the review positive?') is not specific enough. \n\nData:\n"
            # llm_prompt_end = self.get_llm_prompt_suffix()
            llm_prompt_end = "Question:\n"
            print(
                f'calling explain_dataset_iprompt with batch size {self.args.batch_size}')
            prompts, metadata = imodelsx.explain_dataset_iprompt(
                lm=model,
                input_strings=input_strings,
                output_strings=output_strings,
                checkpoint=self.checkpoint,  # which language model to use
                num_learned_tokens=12,  # how long of a prompt to learn
                n_shots=5,  # number of examples in context
                batch_size=self.args.batch_size,  # batch size for iprompt
                llm_float16=False,  # whether to load the model in float_16
                verbose=1,  # how much to print
                prefix_before_input=False, # sets template like ${input}${prefix}
                mask_possible_answers=True,  # only compute loss over valid output tokens
                generation_repetition_penalty=1.0,
                pop_topk_strategy='different_start_token',
                pop_criterion='loss',
                do_final_reranking=True,
                max_n_datapoints=10**5,
                max_length=64, # max length of datapoints (left-truncated before prompt)
                max_n_steps=20, # 200 # limit search by a fixed number of steps
                llm_api="gpt-4",
                # llm_api="gpt-3.5-turbo", # ["gpt-3.5-turbo", "text-curie-001"]
                llm_candidate_regeneration_prompt_start=llm_prompt_start,
                llm_candidate_regeneration_prompt_end=llm_prompt_end,
            )
            
            torch.cuda.empty_cache()
            print('((sent model to cuda))')

            # save stuff
            self.prompt = prompts[0]
            print(f'Got {len(prompts)} prompts. Top prompt: `{prompts[0]}`')
            self.prompts = prompts
            self.meta = metadata

        # set value (calls self.predict)
        self._set_value_acc_samples(model.cuda(), X_text, y)

        return self

    def predict(self, model, X_text: List[str]) -> np.ndarray[int]:
        preds_proba = self.predict_proba(model, X_text)
        return np.argmax(preds_proba, axis=1)

    def predict_proba(self, model, X_text: List[str]) -> np.ndarray[float]:
        target_strs = list(self._get_verbalizer().values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        if self.args.prompt_source == 'data_demonstrations':
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched(
                model=model,
                prompts=[self.prompt + template % (x, '') for x in X_text],
                target_token_ids=target_token_ids,
                batch_size=self.args.batch_size
            )
        else:
            preds = self._get_logit_for_target_tokens_batched(
                model=model,
                prompts=[x + self.prompt for x in X_text],
                target_token_ids=target_token_ids,
                batch_size=self.args.batch_size
            )
        # preds = np.zeros((len(X_text), len(target_token_ids)))
        # for i, x in enumerate(X_text):
        #     preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)
        #     preds[i] = self._get_logit_for_target_tokens(x + self.prompt, target_token_ids)
        assert preds.shape == (len(X_text), len(target_token_ids)), 'preds shape was' + str(
            preds.shape) + ' but should have been ' + str((len(X_text), len(target_token_ids)))

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def _get_logit_for_target_tokens_batched(self,
                                             model,
                                             prompts: List[str],
                                             target_token_ids: List[int],
                                             batch_size: int = 64) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc=f'[{model.device}] getting dataset predictions', colour="red"
        )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start: batch_end]
            self.tokenizer.padding = True
            self.tokenizer.truncation_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = (
                self.tokenizer(
                    prompts_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_attention_mask=True
                )
                .to(model.device)
            )

            # shape is (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                logits = model(**inputs)['logits']
            token_output_positions = inputs['attention_mask'].sum(axis=1)
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append([logits[i, token_output_position, token_output_id].item(
                ) for token_output_id in target_token_ids])

    def _get_first_token_id(self, prompt: str) -> str:
        """Get first token id in prompt
        """
        return self.tokenizer(prompt)['input_ids'][0]

    def _get_verbalizer(self):
        if hasattr(self.args, 'verbalizer') and self.args.verbalizer is not None:
            return self.args.verbalizer
        else:
            return {0: ' Negative.', 1: ' Positive.'}

    def __str__(self):
        return f'PromptStump(val={self.value_mean:0.2f} n={np.sum(self.n_samples)} prompt={self.prompt})'

    def get_str_simple(self):
        return self.prompt


class KeywordStump(Stump):

    def fit(self, model, X_text: List[str], y, feature_names=None, X=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        assert len(np.unique(y)) <= 2, 'only binary classification is supported'
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # fit stump
        if self.split_strategy == 'linear':
            self.stump_keywords_idxs = self._get_stump_keywords_linear(X, y)
        elif self.split_strategy == 'cart':
            self.stump_keywords_idxs = self._get_stump_keywords_cart(X, y)
        self.stump_keywords = self.feature_names[self.stump_keywords_idxs]

        # set value
        self._set_value_acc_samples(X_text, y)
        if self.failed_to_split:
            return self

        # checks
        if self.assert_checks:
            preds_text = self.predict(model, X_text)
            preds_tab = self._predict_tabular(X)
            assert np.all(
                preds_text == preds_tab), 'predicting with text and tabular should give same results'
            assert self.value[1] > self.value[0], 'right child should have greater val than left but value=' + \
                str(self.value)
            assert self.value[1] > self.value_mean, 'right child should have greater val than parent ' + \
                str(self.value)

        return self

    def predict(self, model, X_text: List[str]) -> np.ndarray[int]:
        """Returns prediction 1 for positive and 0 for negative.
        """
        keywords = self.stump_keywords
        ngrams_used_to_predict = max(
            [len(keyword.split(' ')) for keyword in keywords])

        def contains_any_of_keywords(text):
            text = text.lower()
            text = imodelsx.util.generate_ngrams_list(
                text,
                ngrams=ngrams_used_to_predict,
                tokenizer_ngrams=self.tokenizer,
                all_ngrams=True
            )
            for keyword in keywords:
                if keyword in text:
                    return 1
            return 0
        contains_keywords = 1 * \
            np.array([contains_any_of_keywords(x) for x in X_text])
        if self.pos_or_neg == 'pos':
            return contains_keywords
        else:
            return 1 - contains_keywords

    def _predict_tabular(self, X):
        X = imodels.util.arguments.check_fit_X(X)
        # predict whether input has any of the features in stump_keywords_idxs
        X_feats = X[:, self.stump_keywords_idxs]
        pred = np.any(X_feats, axis=1)
        if self.pos_or_neg == 'pos':
            return pred.astype(int)
        else:
            return 1 - pred

    def _get_stump_keywords_linear(self, X, y):
        # fit a linear model
        m = LogisticRegression().fit(X, y)
        m.fit(X, y)

        # find the largest magnitude coefs
        abs_feature_idxs = m.coef_.argsort().flatten()
        bot_feature_idxs = abs_feature_idxs[:self.max_features]
        top_feature_idxs = abs_feature_idxs[-self.max_features:][::-1]

        # return the features with the largest magnitude coefs
        if np.sum(abs(bot_feature_idxs)) > np.sum(abs(top_feature_idxs)):
            self.pos_or_neg = 'neg'
            return bot_feature_idxs
        else:
            self.pos_or_neg = 'pos'
            return top_feature_idxs

    def _get_stump_keywords_cart(self, X, y):
        '''Find the top self.max_features features selected by CART
        '''
        criterion_func = imodelsx.metrics.gini_binary

        # Calculate the gini impurity reduction for each (binary) feature in X
        impurity_reductions = []

        # whether the feature increases the likelihood of the positive class
        feature_positive = []
        y_mean = np.mean(y)
        n = y.size
        gini_impurity = 1 - criterion_func(y_mean)
        for i in range(X.shape[1]):
            x = X[:, i]
            idxs_r = x > 0.5
            idxs_l = x <= 0.5
            if idxs_r.sum() == 0 or idxs_l.sum() == 0:
                impurity_reductions.append(0)
                feature_positive.append(True)
            else:
                y_mean_l = np.mean(y[idxs_l])
                y_mean_r = np.mean(y[idxs_r])
                gini_impurity_l = 1 - criterion_func(y_mean_l)
                gini_impurity_r = 1 - criterion_func(y_mean_r)
                # print('l', indexes_l.sum(), 'r', indexes_r.sum(), 'n', n)
                impurity_reductions.append(
                    gini_impurity
                    - (idxs_l.sum() / n) * gini_impurity_l
                    - (idxs_r.sum() / n) * gini_impurity_r
                )
                feature_positive.append(y_mean_r > y_mean_l)

        impurity_reductions = np.array(impurity_reductions)
        feature_positive = np.arange(X.shape[1])[np.array(feature_positive)]

        # find the top self.max_features with the largest impurity reductions
        args_largest_reduction_first = np.argsort(impurity_reductions)[::-1]
        self.impurity_reductions = impurity_reductions[args_largest_reduction_first][:self.max_features]
        # print('\ttop_impurity_reductions', impurity_reductions[args_largest_reduction_first][:5],
        #   'max', max(impurity_reductions))
        # print(f'\t{X.shape=}')
        imp_pos_top = [
            k for k in args_largest_reduction_first
            if k in feature_positive
            and not k in imodelsx.util.STOPWORDS
        ][:self.max_features]
        imp_neg_top = [
            k for k in args_largest_reduction_first
            if not k in feature_positive
            and not k in imodelsx.util.STOPWORDS
        ][:self.max_features]

        # feat = DecisionTreeClassifier(max_depth=1).fit(X, y).tree_.feature[0]
        if np.sum(imp_pos_top) > np.sum(imp_neg_top):
            self.pos_or_neg = 'pos'
            return imp_pos_top
        else:
            self.pos_or_neg = 'neg'
            return imp_neg_top

    def __str__(self):
        keywords = self.stump_keywords
        keywords_str = ", ".join(keywords[:5])
        if len(keywords) > 5:
            keywords_str += f'...({len(keywords) - 5} more)'
        sign = {'pos': '+', 'neg': '--'}[self.pos_or_neg]
        return f'Stump(val={self.value_mean:0.2f} n={self.n_samples}) {sign} {keywords_str}'

    def get_str_simple(self):
        keywords = self.stump_keywords
        keywords_str = ", ".join(keywords[:5])
        if len(keywords) > 5:
            keywords_str += f'...({len(keywords) - 5} more)'
        sign = {'pos': '+', 'neg': '--'}[self.pos_or_neg]
        return f'{sign} {keywords_str}'
