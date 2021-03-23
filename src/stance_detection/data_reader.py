from typing import List, Union

import pandas as pd
from math import ceil


class Fnc1Reader:
    """Reads the Fake News Detection data set."""

    def __init__(self, loc: str):
        """Inits the data reader with the data at the given location. Expects train and test set data."""
        self.loc = loc
        if self.loc[len(loc) - 1] != '/':
            self.loc += '/'
        self.train_bodies, self.train_stances = self.read_train()
        self.test_bodies, self.test_stances = self.read_test()

    def read_train(self) -> [pd.DataFrame, pd.DataFrame]:
        """ Reads the train set from the data location."""
        bodies = pd.read_csv(self.loc + 'train_bodies.csv', names=['Body ID', 'articleBody'], header=1)
        stances = pd.read_csv(self.loc + 'train_stances.csv', names=['Headline', 'Body ID', 'Stance'], header=1)
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

    def get_body_train(self, body_id: int):
        """Returns the right body text from the train set."""
        return self.train_bodies.loc[self.train_bodies['Body ID'] == body_id]['articleBody'].to_list()[0]


reader = Fnc1Reader('src/data/fnc-1/')
print(reader.kfold(10))
print(reader.get_body_train(10))
