from typing import List
import numpy as np
import imodels
import imodelsx.util
from tprompt.stump import KeywordStump, PromptStump, Stump
import tprompt.data
import logging
import warnings
import copy
import random
from transformers import AutoModelForCausalLM
global model
model = None
class Tree:
    def __init__(
        self,
        args,
        max_depth: int = 3,
        split_strategy: str='iprompt',
        verbose=True,
        tokenizer=None,
        assert_checks=True,
        checkpoint: str='EleutherAI/gpt-j-6B',
        checkpoint_prompting: str='EleutherAI/gpt-j-6B',        
        device='cuda',
    ):
        '''
        Params
        ------
        max_depth: int
            Maximum depth of the tree.
        split_strategy: str
            'manual' - use passed prompts in args.prompts_list
            'iprompt' - use prompted language model to split
            'cart' - use cart to split
            'linear' - use linear to split
        verbose: bool
        tokenizer
        assert_checks: bool
            Whether to run checks during fitting
        checkpoint: str
            the underlying model used for prediction
        checkpoint_prompting: str
            the model used for finding the prompt            
        '''
        self.args = args
        self.max_depth = max_depth
        self.split_strategy = split_strategy
        self.verbose = verbose
        self.assert_checks  = assert_checks
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        self.device = device
        if tokenizer is None:
            self.tokenizer = imodelsx.util.get_spacy_tokenizer(convert_output=False)
        else:
            self.tokenizer = tokenizer
        self.prompts_list = []

    def fit(self, X_text: List[str]=None, y=None, feature_names=None, X=None):
        if X is None and X_text:
            pass
            #warnings.warn("X is not passed, defaulting to generating unigrams from X_text")
            #X, _, feature_names = tprompt.data.convert_text_data_to_counts_array(X_text, [], ngrams=1)

        # check and set some attributes
        #X, y, _ = imodels.util.arguments.check_fit_arguments(
        #    self, X, y, feature_names)
        if isinstance(X_text, list):
            X_text = np.array(X_text).flatten()
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        global model
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint).cuda()
        # set up arguments
        stump_kwargs = dict(
            args=self.args,
            tokenizer=self.tokenizer,
            split_strategy=self.split_strategy,
            assert_checks=self.assert_checks,
            verbose=self.verbose,
            model=model,
            checkpoint=self.checkpoint,
            checkpoint_prompting=self.checkpoint_prompting,
        )
        if self.split_strategy in ['iprompt']:
            stump_class = PromptStump
        else:
            stump_class = KeywordStump


        # fit root stump
        # if the initial feature puts no points into a leaf,
        # the value will end up as NaN
        X_text_all = X_text
        X_text_all = []
        for text in X_text:
            X_text_all.append(f'{text}\nOutput:')
        #import pdb; pdb.set_trace()
        y_all = y
        X_all = X

        num = 1000
        X_text_train = X_text_all[:num]
        X_text = X_text_all[num:]
        y_train = y_all[:num]
        y = y_all[num:]
        #X_train = X_all[:num]
        #X = X_all[num:]

        #num_positive = 10
        #num_negative = num_positive
        #stump_kwargs['verbalizer'] = 
        stump = stump_class(**stump_kwargs)
        verbalizer_dict = stump._get_verbalizer()
        input_strings_train = X_text_train
        #output_strings_train = [verbalizer_dict[int(yi)] for yi in y_train]
        #output_strings_train = [verbalizer_dict[int(yi)] for yi in y_train]
        prompts = []
        unique_ys = sorted(list(set(y_train)), key=lambda x: -x) # 1, 0 since positive usually comes first
        examples_by_y = {}
        for y_i in unique_ys:
            examples_by_y[y_i] = sorted(list(filter(lambda ex: ex[1]==y_i, zip(X_text_train, y_train))))
        prompts = []
        num_prompts = 9
        num_prompts = 25
        num_prompts = self.args.num_prompts
        num_prompts_max = self.args.num_prompts_max
        num_repeat = 1
        num_repeat = self.args.num_repeat
        #num_prompts = 100
        print (f'Num prompts max {num_prompts_max}')
        print (f'Num prompts {num_prompts}')
        #import pdb; pdb.set_trace()
        print (f'Repeating {num_repeat}')
        if self.args.rest_negative == 1:
            print ('REST NEG')
        while len(prompts) < num_prompts_max:
            #prompt = ''
            prompt = []
            unique_ys_copy = copy.deepcopy(unique_ys)
            random.shuffle(unique_ys_copy)

            for _ in range(num_repeat):
                for iii, y_i in enumerate(unique_ys_copy[:2]):
                    if iii == 0 and self.args.rest_negative == 1:
                        y_i = random.choice(unique_ys_copy[:iii+1] + unique_ys_copy[iii+2:])
                    example = random.choice(examples_by_y[y_i])
                    text, _ = example
                    #prompt += f'Input: {text}{verbalizer_dict[iii]}\n'
                    prompt.append(f'Input: {text}{verbalizer_dict[iii]}\n')
            random.shuffle(prompt)
            #prompt = 'Find the pattern between the input and the output.\n' + ''.join(prompt)
            prompt = ''.join(prompt)
            prompt += 'Input: '
            if prompt not in prompts:
                prompts.append(prompt)

        ###positive_examples = []
        ###negative_examples = []
        #import pdb; pdb.set_trace()
        ###for input_string, output_string in zip(input_strings_train, output_strings_train):
        ###    if output_string == verbalizer_dict[1]:
        ###        if len(positive_examples) < num_positive:
        ###            positive_examples.append((input_string, output_string))
        ###    elif output_string == verbalizer_dict[0]:
        ###        if len(negative_examples) < num_negative:
        ###            negative_examples.append((input_string, output_string))
        ###    else:
        ###        assert False
        ####import pdb; pdb.set_trace()
        ###prompts = []
        ###for positive_example in positive_examples:
        ###    input_string_pos, output_string_pos = positive_example
        ###    for negative_example in negative_examples:
        ###        input_string_neg, output_string_neg = negative_example
        ###        prompt = f'Input: {input_string_pos}{output_string_pos}\nInput: {input_string_neg}{output_string_neg}\nInput: '
        ###        prompts.append(prompt)
        ####import pdb; pdb.set_trace()
        num = 4096
        #num = 64
        #num = 256
        #num = 1024
        X_text = X_text[:num]
        y = y[:num]
        #X = X[:num]


        stump = stump_class(**stump_kwargs).fit(
            X_text=X_text,
            y=y,
            feature_names=self.feature_names,
            X=X,
            prompts=prompts
        )
        #stump.idxs = np.ones(X.shape[0], dtype=bool)
        stump.idxs = np.ones(len(X_text), dtype=bool)
        self.root_ = stump

        # recursively fit stumps and store as a decision tree
        stumps_queue = [stump]
        i = 0
        depth = 1
        #import pdb; pdb.set_trace()
        while depth < self.max_depth:
            stumps_queue_new = []
            for stump in stumps_queue:
                stump = stump
                if self.verbose:
                    logging.debug(f'Splitting on {depth=} stump_num={i} {stump.idxs.sum()=}')
                #idxs_pred = stump.predict(X_text=X_text) > 0.5
                idxs_pred = stump.predict_split(X_text=X_text) > 0.5
                #import pdb; pdb.set_trace()
                for idxs_p, attr in zip([~idxs_pred, idxs_pred], ['child_left', 'child_right']):
                    # for idxs_p, attr in zip([idxs_pred], ['child_right']):
                    idxs_child = stump.idxs & idxs_p
                    if self.verbose:
                        logging.debug(f'\t{idxs_pred.sum()=} {idxs_child.sum()=}', len(np.unique(y[idxs_child])))
                    if idxs_child.sum() > 0 \
                        and idxs_child.sum() < stump.idxs.sum() \
                            and len(np.unique(y[idxs_child])) > 1:

                        # fit a potential child stump
                        #import pdb; pdb.set_trace()
                        X_text_child = []
                        for text, idx_child in zip(X_text, idxs_child):
                            if idx_child:
                                X_text_child.append(text)
                        #import pdb; pdb.set_trace()
                        stump_child = stump_class(**stump_kwargs).fit(
                            X_text=X_text_child,
                            y=y[idxs_child],
                            feature_names=self.feature_names,
                            #X=X[idxs_child],
                            prompts=prompts
                        )

                        # make sure the stump actually found a non-trivial split
                        if not stump_child.failed_to_split:
                            stump_child.idxs = idxs_child
                            #acc_tree_baseline = np.mean(self.predict_label(
                            #    X_text=X_text[idxs_child]) == y[idxs_child])
                            if attr == 'child_left':
                                stump.child_left = stump_child
                            else:
                                stump.child_right = stump_child
                            stumps_queue_new.append(stump_child)
                            #if self.verbose:
                            #    logging.debug(f'\t\t {stump.stump_keywords} {stump.pos_or_neg}')
                            i += 1

                        ######################### checks ###########################
                            self.assert_checks = False
                            if self.assert_checks:
                                # check acc for the points in this stump
                                acc_tree = np.mean(self.predict(
                                    X_text=X_text[idxs_child]) == y[idxs_child])
                                assert acc_tree >= acc_tree_baseline, f'stump acc {acc_tree:0.3f} should be > after adding child {acc_tree_baseline:0.3f}'

                                # check total acc
                                acc_total_baseline = max(y.mean(), 1 - y.mean())
                                acc_total = np.mean(self.predict(X_text=X_text) == y)
                                assert acc_total >= acc_total_baseline, f'total acc {acc_total:0.3f} should be > after adding child {acc_total_baseline:0.3f}'

                                # check that stumptrain acc improved over this set
                                # not necessarily going to improve total acc, since the stump always predicts 0/1
                                # even though the correct answer might be always 0 or always be 1
                                acc_child_baseline = min(
                                    y[idxs_child].mean(), 1 - y[idxs_child].mean())
                                assert stump_child.acc > acc_child_baseline, f'acc {stump_child.acc:0.3f} should be > baseline {acc_child_baseline:0.3f}'


            stumps_queue = stumps_queue_new
            depth += 1
        
        if isinstance(self.root_, PromptStump):
            self._set_prompts_list(self.root_)

        return self

    def predict_proba_old(self, X_text: List[str] = None):
        #import pdb; pdb.set_trace()
        preds = []
        for x_t in X_text:

            # prediction for single point
            stump = self.root_
            while stump:
                # 0 or 1 class prediction here
                pred = stump.predict(X_text=[x_t])[0]
                value = stump.value

                if pred > 0.5:
                    stump = stump.child_right
                    value = value[1]
                else:
                    stump = stump.child_left
                    value = value[0]

                if stump is None:
                    preds.append(value)
        preds = np.array(preds)
        probs = np.vstack((1 - preds, preds)).transpose()  # probs (n, 2)
        return probs

    def predict_proba(self, X_text: List[str] = None):
        #import pdb; pdb.set_trace()
        preds = []
        for x_t in X_text:
            # prediction for single point
            stump = self.root_
            while stump:
                # 0 or 1 class prediction here
                pred_split = stump.predict_split(X_text=[x_t + '\nOutput:'])
                #value = stump.value

                if pred_split == 0:
                    value = stump.best_left_label
                    stump = stump.child_left
                else:
                    value = stump.best_right_label
                    stump = stump.child_right

                if stump is None:
                    preds.append(value)
        preds = np.array(preds)
        #probs = np.vstack((1 - preds, preds)).transpose()  # probs (n, 2)
        return preds

    def predict(self, X_text: List[str] = None) -> np.ndarray[int]:
        #import pdb; pdb.set_trace()
        preds_bool = self.predict_proba(X_text=X_text)
        return preds_bool.astype(int)
    

    def __str__(self):
        s = f'> Tree(max_depth={self.max_depth} split_strategy={self.split_strategy})\n> ------------------------------------------------------\n'
        return s + self.viz_tree()

    def viz_tree(self, stump: Stump=None, depth: int=0, s: str='') -> str:
        if stump is None:
            stump = self.root_
        s += '   ' * depth + str(stump) + '\n'
        if stump.child_left:
            s += self.viz_tree(stump.child_left, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Neg n={stump.n_samples[0]} val={stump.value[0]:0.3f}' + '\n'
        if stump.child_right:
            s += self.viz_tree(stump.child_right, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Pos n={stump.n_samples[1]} val={stump.value[1]:0.3f}' + '\n'
        return s

    def _set_prompts_list(self, stump: Stump) -> List[str]:
        self.prompts_list.append(stump.prompt)
        if stump.child_left:
            self._set_prompts_list(stump.child_left)
        if stump.child_right:
            self._set_prompts_list(stump.child_right)
        return self.prompts_list
