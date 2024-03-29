from typing import List
import numpy as np
import imodels
import imodelsx.util
from tprompt.stump import PromptStump
import tprompt.data
import logging
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from tprompt.utils import load_lm


class Tree:
    def __init__(
        self,
        args,
        max_depth: int = 3,
        split_strategy: str = "iprompt",
        verbose=True,
        tokenizer=None,
        assert_checks=True,
        checkpoint: str = "EleutherAI/gpt-j-6B",
        checkpoint_prompting: str = "EleutherAI/gpt-j-6B",
        device="cuda",
    ):
        """
        Params
        ------
        max_depth: int
            Maximum depth of the tree.
        split_strategy: str
            'manual' - use passed prompts in args.prompts_list
            'iprompt' - use iprompt to split
        verbose: bool
        tokenizer
        assert_checks: bool
            Whether to run checks during fitting
        checkpoint: str
            the underlying model used for prediction
        checkpoint_prompting: str
            the model used for finding the prompt
        """
        self.args = args
        self.max_depth = max_depth
        self.split_strategy = split_strategy
        self.verbose = verbose
        self.assert_checks = assert_checks
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        self.device = device
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.tokenizer = tokenizer
        self.prompts_list = []

    def fit(self, X_text: List[str] = None, y=None, feature_names=None):

        # check and set some attributes
        if isinstance(X_text, list):
            X_text = np.array(X_text).flatten()
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # set up arguments
        model = load_lm(checkpoint=self.checkpoint, tokenizer=self.tokenizer)
        stump_kwargs = dict(
            args=self.args,
            tokenizer=self.tokenizer,
            split_strategy=self.split_strategy,
            assert_checks=self.assert_checks,
            verbose=self.verbose,
            model=model,
            checkpoint=self.checkpoint,
            checkpoint_prompting=self.checkpoint_prompting,
            batch_size=self.args.batch_size,
        )

        # fit root stump
        # if the initial feature puts no points into a leaf,
        # the value will end up as NaN
        stump = PromptStump(**stump_kwargs).fit(
            X_text=X_text, y=y, feature_names=self.feature_names
        )
        stump.idxs = np.ones(len(X_text), dtype=bool)
        self.root_ = stump

        # recursively fit stumps and store as a decision tree
        if stump.failed_to_split:
            stumps_queue = []
            warnings.warn("Failed to split on root stump!")
        else:
            stumps_queue = [stump]
        i = 0
        depth = 1
        while depth < self.max_depth:
            stumps_queue_new = []
            for stump in stumps_queue:
                stump = stump
                if self.verbose:
                    logging.debug(
                        f"Splitting on {depth=} stump_num={i} {stump.idxs.sum()=}"
                    )
                idxs_pred = stump.predict(X_text=X_text) > 0.5
                for idxs_p, attr in zip(
                    [~idxs_pred, idxs_pred], ["child_left", "child_right"]
                ):
                    # for idxs_p, attr in zip([idxs_pred], ['child_right']):
                    idxs_child = stump.idxs & idxs_p
                    if self.verbose:
                        logging.debug(
                            f"\t{idxs_pred.sum()=} {idxs_child.sum()=}",
                            len(np.unique(y[idxs_child])),
                        )
                    if (
                        idxs_child.sum() > 0
                        and idxs_child.sum() < stump.idxs.sum()
                        and len(np.unique(y[idxs_child])) > 1
                    ):
                        # fit a potential child stump
                        stump_child = PromptStump(**stump_kwargs).fit(
                            X_text=X_text[idxs_child],
                            y=y[idxs_child],
                            feature_names=self.feature_names,
                        )

                        # make sure the stump actually found a non-trivial split
                        if not stump_child.failed_to_split:
                            stump_child.idxs = idxs_child
                            acc_tree_baseline = np.mean(
                                self.predict(X_text=X_text[idxs_child]) == y[idxs_child]
                            )
                            if attr == "child_left":
                                stump.child_left = stump_child
                            else:
                                stump.child_right = stump_child
                            stumps_queue_new.append(stump_child)
                            i += 1

                            ######################### checks ###########################
                            if self.assert_checks:
                                # check acc for the points in this stump
                                acc_tree = np.mean(
                                    self.predict(X_text=X_text[idxs_child])
                                    == y[idxs_child]
                                )
                                assert (
                                    acc_tree >= acc_tree_baseline
                                ), f"stump acc {acc_tree:0.3f} should be > after adding child {acc_tree_baseline:0.3f}"

                                # check total acc
                                acc_total_baseline = max(y.mean(), 1 - y.mean())
                                acc_total = np.mean(self.predict(X_text=X_text) == y)
                                assert (
                                    acc_total >= acc_total_baseline
                                ), f"total acc {acc_total:0.3f} should be > after adding child {acc_total_baseline:0.3f}"

                                # check that stumptrain acc improved over this set
                                # not necessarily going to improve total acc, since the stump always predicts 0/1
                                # even though the correct answer might be always 0 or always be 1
                                acc_child_baseline = min(
                                    y[idxs_child].mean(), 1 - y[idxs_child].mean()
                                )
                                assert (
                                    stump_child.acc > acc_child_baseline
                                ), f"acc {stump_child.acc:0.3f} should be > baseline {acc_child_baseline:0.3f}"

            stumps_queue = stumps_queue_new
            depth += 1

        if isinstance(self.root_, PromptStump):
            self._set_prompts_list(self.root_)

        return self

    def predict_proba(self, X_text: List[str] = None):
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

    def predict(self, X_text: List[str] = None) -> np.ndarray[int]:
        preds_bool = self.predict_proba(X_text=X_text)[:, 1]
        return (preds_bool > 0.5).astype(int)

    def __str__(self):
        s = f"> Tree(max_depth={self.max_depth} split_strategy={self.split_strategy})\n> ------------------------------------------------------\n"
        return s + self.viz_tree()

    def viz_tree(self, stump: PromptStump= None, depth: int = 0, s: str = "") -> str:
        if stump is None:
            stump = self.root_
        s += "   " * depth + str(stump) + "\n"
        if stump.child_left:
            s += self.viz_tree(stump.child_left, depth + 1)
        else:
            s += (
                "   " * (depth + 1)
                + f"Neg n={stump.n_samples[0]} val={stump.value[0]:0.3f}"
                + "\n"
            )
        if stump.child_right:
            s += self.viz_tree(stump.child_right, depth + 1)
        else:
            s += (
                "   " * (depth + 1)
                + f"Pos n={stump.n_samples[1]} val={stump.value[1]:0.3f}"
                + "\n"
            )
        return s

    def _set_prompts_list(self, stump: PromptStump) -> List[str]:
        self.prompts_list.append(stump.prompt)
        if stump.child_left:
            self._set_prompts_list(stump.child_left)
        if stump.child_right:
            self._set_prompts_list(stump.child_right)
        return self.prompts_list


class TreeRegressor(Tree, RegressorMixin):
    ...


class TreeClassifier(Tree, ClassifierMixin):
    ...
