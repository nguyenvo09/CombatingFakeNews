"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import torch_utils
import time
np.random.seed(123456)

class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()
        self._candidate = dict() # negative candidates

    def set_interactions(self, interactions):
        """

        Parameters
        ----------
        interactions: :class:`interactions.Interactions`


        Returns
        -------

        """
        csr_data = interactions.tocsr()
        self.build_neg_dict(csr_data)

    def build_neg_dict(self, csr_data):
        #for each user, store the unobserved values into a dict for sampling later.
        # csr_data = csr_matrix(csr_data)
        # n_users, n_items = csr_data.shape
        # user_counts = np.zeros(n_users)
        # for u in range(n_users): user_counts = csr_data[u].getnnz()
        pass

    def random_sample_items(self, num_items, shape, random_state=None):
        """
        Randomly sample a number of items based on shape.
        (we need to improve this since it is likely to sample a positive instance)
        https://github.com/maciejkula/spotlight/issues/36
        https://github.com/graytowne/caser_pytorch/blob/master/train_caser.py
        Parameters
        ----------

        num_items: int
            Total number of items from which we should sample:
            the maximum value of a sampled item id will be smaller
            than this.
        shape: int or tuple of ints
            Shape of the sampled array.
        random_state: np.random.RandomState instance, optional
            Random state to use for sampling.

        Returns
        -------

        items: np.array of shape [shape]
            Sampled item ids.
        """

        if random_state is None:
            random_state = np.random.RandomState()

        items = random_state.randint(0, num_items, shape, dtype = np.int64)
        # items = random_state.randint(1, num_items, shape, dtype=np.int64) #random from 1 to num_items as 0 is PADDING_IDX

        return items

    # reuse from https://github.com/nguyenvo09/caser_pytorch/blob/master/train_caser.py#L203
    def get_train_instances(self, interactions, num_negatives, random_state=None):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}
        Parameters
        ----------
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        num_negatives: int
            total number of negatives to sample for each sequence
        """
        if random_state is None:
            random_state = np.random.RandomState()
        user_ids = interactions.user_ids.astype(np.int64) # may not be unique
        item_ids = interactions.item_ids.astype(np.int64)
        negative_samples = np.zeros((user_ids.shape[0], num_negatives), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items)
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(user_ids):
            for j in range(num_negatives):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    random_state.randint(len(x))]

        return user_ids, item_ids, negative_samples