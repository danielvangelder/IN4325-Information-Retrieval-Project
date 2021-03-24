from typing import List, Union

import pandas as pd
from math import ceil
import numpy as np


class Fnc1Reader:
    """Reads the Fake News Detection data set."""

    def __init__(self, loc: str):
        """Inits the data reader with the data at the given location. Expects train and test set data."""
        self.loc = loc
        if self.loc[len(loc) - 1] != '/':
            self.loc += '/'
        self.train_bodies, self.train_stances = self.read_train()
        self.test_bodies, self.test_stances = self.read_test()
        self.comp_bodies, self.comp_stances = self.read_comp()

    def read_train(self) -> [pd.DataFrame, pd.DataFrame]:
        """Reads the train set from the data location."""
        return self.read_labelled('train_bodies.csv', 'train_stances.csv')

    def read_comp(self) -> [pd.DataFrame, pd.DataFrame]:
        """Reads the competition data set from the data location"""
        return self.read_labelled('competition_test_bodies.csv', 'competition_test_stances.csv')

    def read_labelled(self, bodies_loc: str, stances_loc: str) -> [pd.DataFrame, pd.DataFrame]:
        bodies = pd.read_csv(self.loc + bodies_loc, names=['Body ID', 'articleBody'], header=1)
        stances = pd.read_csv(self.loc + stances_loc, names=['Headline', 'Body ID', 'Stance'], header=1)
        labels = list(map(self.stance_to_label, stances['Stance'].to_list()))
        stances['Label'] = labels
        assert len(bodies) != 0 and len(stances) != 0
        assert bodies.columns.to_list() == ['Body ID', 'articleBody'] \
               and stances.columns.to_list() == ['Headline', 'Body ID', 'Stance', 'Label']

        return bodies, stances

    def stance_to_label(self, stance: str) -> int:
        """
        1, Agrees: The body text agrees with the headline.
        2, Disagrees: The body text disagrees with the headline.
        3, Discusses: The body text discuss the same topic as the headline, but does not take a position
        4, Unrelated: The body text discusses a different topic than the headline
        """
        if stance == 'agree':
            return 1
        elif stance == 'disagree':
            return 2
        elif stance == 'discuss':
            return 3
        elif stance == 'unrelated':
            return 4
        raise Exception('Stance does not exist: ' + stance)

    def read_test(self) -> [pd.DataFrame, pd.DataFrame]:
        """Reads the test set from the data location."""
        bodies = pd.read_csv(self.loc + 'train_bodies.csv', names=['Body ID', 'articleBody'], header=1)
        stances = pd.read_csv(self.loc + 'train_stances.csv', names=['Headline', 'Body ID'], header=1)
        assert len(bodies) != 0 and len(stances) != 0
        assert bodies.columns.to_list() == ['Body ID', 'articleBody'] \
               and stances.columns.to_list() == ['Headline', 'Body ID']

        return bodies, stances

    def kfold(self, n: int) -> List[pd.DataFrame]:
        """Returns a list of n random folds of the training set."""
        size = len(self.train_stances.index)
        shuffled = self.train_stances.sample(size)
        folds = []

        for i in range(0, n - 1):
            lower = ceil(i / n * size)
            upper = ceil((i + 1) / n * size)
            if i == n - 1:
                upper = size
            fold = shuffled.iloc[lower:upper]
            folds.append(fold)

        return folds

    def get_body_train(self, body_id: int) -> str:
        """Returns the right body text from the train set."""
        return self.train_bodies.loc[self.train_bodies['Body ID'] == body_id]['articleBody'].to_list()[0]

    def evaluate_comp(self, labels: Union[List[int], List[str]]) -> float:
        """Evaluates the given labels on the competition data set."""
        if all(isinstance(label, int) for label in labels):
            return self.evaluate_fold(self.comp_stances, labels)
        elif all(isinstance(label, str) for label in labels):
            return self.evaluate_fold(self.comp_stances, list(map(self.stance_to_label, labels)))
        else:
            raise Exception('Bad labels format: ' + str(type(labels)))

    def evaluate_fold(self, fold: pd.DataFrame, labels: List[int]) -> float:
        """Evaluates a data fold with the given labels"""
        assert len(fold.index) == len(labels)
        score = 0
        for i, row in fold.iterrows():
            score += self.score(row['Label'], labels[i])
        return score

    def score(self, actual: int, output: int) -> float:
        """
        As in scorer.py provided by FNC-1.
        +0.25 for each correct unrelated
        +0.25 for each correct related (label is any of agree, disagree, discuss)
        +0.75 for each correct agree, disagree, discuss
        """
        assert output in [1, 2, 3, 4]
        score = 0
        if actual == output:
            score += 0.25
            if actual != 4:
                score += 0.50
        if actual in [1, 2, 3] and output in [1, 2, 3]:
            score += 0.25
        return score


reader = Fnc1Reader('src/data/fnc-1/')
# print(reader.kfold(10))
# print(reader.get_body_train(10))
labels: List[int] = np.random.randint(1, 5, len(reader.comp_stances.index)).tolist()
print(reader.evaluate_comp(labels))
