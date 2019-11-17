import torch
import numpy as np
import torch_utils
import nets as my_nets
import losses as my_losses
import torch_utils as my_utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from handlers import output_handler, sampler
from Evaluation import evaluator as my_evaluator
import datetime
import json
class MF_model(object):
    """
    Model for MF.

    Parameters
    ----------

    loss: string, optional
        The loss function for approximating a softmax with negative sampling.
        One of 'pointwise', 'bpr', 'hinge', 'adaptive_hinge', corresponding
        to losses from :class:`spotlight.losses`.

    embedding_dim: int, optional
        Number of embedding dimensions to use for representing items.
        Overridden if representation is an instance of a representation class.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    num_negative_samples: int, optional
        Number of negative samples to generate for adaptive hinge loss.

    """
    def __init__(self,
                 loss = 'pointwise',
                 embedding_dim = 32,
                 n_iter = 10,
                 batch_size = 256,
                 reg_l2 = 1e-6,         # L2 norm, followed Caser
                 learning_rate = 1e-3, # learning rate or step size, followed Caser
                 decay_step = 500,
                 decay_weight = 0.1,
                 layers_size = [64, 32, 16, 8],  # layers' size
                 optimizer_func = None, # e.g. adam
                 use_cuda = False,
                 # sparse = False,
                 random_state = None,
                 num_negative_samples = 3,
                 trained_net = None,
                 net_type = "mf",
                 logfolder = None):
        assert loss in ('pointwise', 'bpr', 'hinge', 'adaptive_hinge')
        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._reg_l2 = reg_l2
        # self._decay_step = decay_step
        # self._decay_weight = decay_weight

        self._layers_size = layers_size
        self._optimizer_func = optimizer_func

        self._use_cuda = use_cuda
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = num_negative_samples

        self._n_users, self._n_items = None, None
        self._net = None
        self._optimizer = None
        # self._lr_decay = None
        self._loss_func = None
        self._net_type = net_type
        assert logfolder != ""
        self.logfolder = logfolder
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        curr_date = datetime.datetime.now().timestamp() # second
        self.logfile_text = os.path.join(logfolder, "%s_result.txt" % int(curr_date))
        self.saved_model = os.path.join(logfolder, "%s_saved_model" % int(curr_date))
        self.output_handler = output_handler.FileHandler(self.logfile_text)

        # for re-producing experiments, adopted from spotlight
        my_utils.set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                             cuda = self._use_cuda)

        # for evaluation during training
        self._sampler = sampler.Sampler()
        self._candidate = dict()

        if trained_net is not None:
            self._net = trained_net

    def __repr__(self):
        """ Return a string of the model when you want to print"""
        # todo
        return "Vanilla Matrix Factorization Model"

    def _initialized(self):

        return self._net is not None

    def _initialize(self, interactions):
        """

        Parameters
        ----------
        interactions: :class:`interactions.Interactions`
        Returns
        -------

        """
        self._n_users, self._n_items = interactions.num_users, interactions.num_items
        if self._net_type == "mf":
            self._net = my_nets.MF(self._n_users, self._n_items, self._embedding_dim)

        # put the model into cuda if use cuda
        self._net = my_utils.gpu(self._net, self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay = self._reg_l2,
                lr = self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        # losses functions
        if self._loss == 'pointwise':
            self._loss_func = my_losses.pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = my_losses.bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = my_losses.hinge_loss
        elif self._loss == 'bce':  # binary cross entropy
            self._loss_func = my_losses.pointwise_bceloss
        else:
            self._loss_func = my_losses.adaptive_hinge_loss

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._n_users:
            raise ValueError('Maximum user id greater than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._n_items:
            raise ValueError('Maximum item id greater than number of items in model.')


    def fit(self, interactions,
            verbose=True,
            topN = 10,
            vadRatings = None,
            vadNegatives = None,
            testRatings = None,
            testNegatives = None,
            adjNetwork = None):
        """
        Fit the model.
        Parameters
        ----------

        interactions: :class:`interactions.Interactions`
            The input sequence dataset.
        vadRatings: :class:`list[list[int]]`
        vadNegatives: :class:`list[list[int]]`
        testRatings: :class:`list[list[int]]`
        testNegatives: :class:`list[list[int]]`
            Negative samples of every pair of (user, item) in  testRatings. shape (bs, 100)
            100 negative samples
        """

        self._sampler.set_interactions(interactions)

        # user_ids = interactions.user_ids.astype(np.int64)
        # item_ids = interactions.item_ids.astype(np.int64)

        if not self._initialized():
            self._initialize(interactions)

        # self._check_input(user_ids, item_ids)

        best_hit = 0
        best_ndcg = 0
        best_epoch = 0
        test_ndcg = 0
        test_hit = 0
        test_results_dict = None


        for epoch_num in range(self._n_iter):

            # ------ Move to here ----------------------------------- #

            user_ids, item_ids, neg_items_ids = self._sampler.get_train_instances(interactions,
                                                                                  self._num_negative_samples,
                                                                                  random_state = self._random_state)
            self._check_input(user_ids, item_ids)
            users, items, neg_items = my_utils.shuffle(user_ids, item_ids, neg_items_ids,
                                                       random_state = self._random_state)

            user_ids_tensor = my_utils.gpu(torch.from_numpy(users), self._use_cuda)
            item_ids_tensor = my_utils.gpu(torch.from_numpy(items), self._use_cuda)
            neg_item_ids_tensor = my_utils.gpu(torch.from_numpy(neg_items), self._use_cuda)

            self._check_shape(user_ids_tensor, item_ids_tensor, neg_item_ids_tensor, self._num_negative_samples)

            epoch_loss = 0.0
            t1 = time.time()
            for (minibatch_num,
                 (batch_user,
                  batch_item,
                  batch_negatives)) in enumerate(my_utils.minibatch(user_ids_tensor,
                                                                    item_ids_tensor,
                                                                    neg_item_ids_tensor,
                                                                    batch_size=self._batch_size)):

                if self._loss == 'adaptive_hinge':
                    positive_prediction, \
                    negative_prediction = self._get_multiple_negative_predictions_adaptive(batch_user, batch_item,
                                                                                            batch_negatives,
                                                                                            self._num_negative_samples)
                else:
                    # need to duplicate batch_user and batch_item
                    positive_prediction, \
                    negative_prediction = self._get_multiple_negative_predictions_normal(batch_user, batch_item,
                                                                                          batch_negatives,
                                                                                          self._num_negative_samples)

                self._optimizer.zero_grad()

                loss = self._loss_func(positive_prediction, negative_prediction)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1
            t2 = time.time()
            epoch_train_time = t2 - t1
            if verbose: # validation after each epoch
                t1 = time.time()
                result_val = my_evaluator.evaluate(self, vadRatings, vadNegatives, topN)
                hits = result_val["hits"]
                ndcg = result_val["ndcg"]

                result_test = my_evaluator.evaluate(self, testRatings, testNegatives, topN)
                hits_test = result_test["hits"]
                ndcg_test = result_test["ndcg"]

                t2 = time.time()
                eval_time = t2 - t1
                self.output_handler.myprint('|Epoch %d | Train time: %d | Train loss: %.3f | Eval time: %.3f (s) '
                      '| Vad hits@%d = %.3f | Vad ndcg@%d = %.3f '
                      '| Test hits@%d = %.3f | Test ndcg@%d = %.3f |'
                      % (epoch_num, epoch_train_time, epoch_loss, eval_time, topN, hits, topN, ndcg, topN, hits_test, topN,
                         ndcg_test))
                if hits > best_hit or (hits == best_hit and ndcg > best_ndcg):
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net, f)
                    test_results_dict = result_test
                    best_hit, best_ndcg, best_epoch = hits, ndcg, epoch_num
                    test_hit, test_ndcg = hits_test, ndcg_test

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))

        self.output_handler.myprint('Best result: '
              '| vad hits@%d = %.3f | vad ndcg@%d = %.3f '
              '| test hits@%d = %.3f | test ndcg@%d = %.3f | epoch = %d' % (topN, best_hit, topN, best_ndcg,
                                                                            topN, test_hit, topN, test_ndcg,
                                                                            best_epoch))
        self.output_handler.myprint_details(json.dumps(test_results_dict, sort_keys = True, indent = 2))

    def _check_shape(self, users, items, neg_items, num_negatives):
        assert users.shape == items.shape
        assert neg_items.shape == (users.shape[0], num_negatives) # key difference

    def _get_negative_prediction(self, user_ids):
        """ Code from Spotlight """
        negative_items = self._sampler.random_sample_items(
            self._n_items,
            len(user_ids),
            random_state=self._random_state)
        negative_var = my_utils.gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._net(user_ids, negative_var)
        return negative_prediction


    def _get_multiple_negative_predictions_adaptive(self, user_ids, item_ids, neg_item_ids, n):
        """
            This function is for adaptive hinge loss.

        Parameters
        ----------
        user_ids: :class:`torch.Tensor`
            shape (batch_size, )
        item_ids: :class:`torch.Tensor`
            shape (batch_size, )
        neg_item_ids: :class:`torch.Tensor`
            shape (batch_size, n)
        n: int

        Returns
        -------

        """
        batch_size = user_ids.size(0)
        assert neg_item_ids.size() == (batch_size, n)
        # needs to check
        user_ids_tmp = user_ids.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)

        assert user_ids_tmp.size() == (batch_size * n,)
        batch_negatives_tmp = neg_item_ids.view(batch_size * n)

        negative_prediction = self._net(user_ids_tmp, batch_negatives_tmp)
        negative_prediction = negative_prediction.view(batch_size, n) # very important
        negative_prediction = negative_prediction.permute(1, 0) # (n, batch_size)

        assert negative_prediction.shape == (n, batch_size)
        positive_prediction = self._net(user_ids, item_ids)  # (batch_size)

        return positive_prediction, negative_prediction

    def _get_multiple_negative_predictions_normal(self, user_ids, item_ids, neg_item_ids, n):
        """
        We compute prediction for every pair of (user, neg_item). Since shape of user_ids is (batch_size, )
        and neg_item_ids.shape = (batch_size, n), we need to reshape user_ids a little bit.

        Parameters
        ----------
        user_ids: :class:`torch.Tensor`
            shape (batch_size, )
        item_ids: :class:`torch.Tensor`
            shape (batch_size, )
        neg_item_ids: :class:`torch.Tensor`
            shape (batch_size, n)
        n: int

        Returns
        -------

        """
        batch_size = user_ids.size(0)
        assert neg_item_ids.size() == (batch_size, n)
        # needs to check
        user_ids_tmp = user_ids.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)

        assert user_ids_tmp.size() == (batch_size * n, )
        batch_negatives_tmp = neg_item_ids.view(batch_size * n)

        negative_prediction = self._net(user_ids_tmp, batch_negatives_tmp)
        positive_prediction = self._net(user_ids, item_ids) # (batch_size)
        positive_prediction = positive_prediction.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)

        assert positive_prediction.shape == negative_prediction.shape
        return positive_prediction, negative_prediction

    def predict(self, user_ids, item_ids):
        """
        Make predictions: given a sequence of interactions, predict
        the next item in the sequence.

        Parameters
        ----------

        sequences: array, (1 x max_sequence_length)
            Array containing the indices of the items in the sequence.
        item_ids: array (num_items x 1), optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: array
            Predicted scores for all items in item_ids.
        """

        self._net.train(False) # very important

        user_ids, item_ids = my_utils._predict_process_ids(user_ids, item_ids, self._n_items, self._use_cuda)
        assert user_ids.shape == item_ids.shape
        out = self._net(user_ids, item_ids)

        return my_utils.cpu(out).detach().numpy().flatten()
