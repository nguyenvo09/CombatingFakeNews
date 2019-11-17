"""
Loss functions for recommender models.

The pointwise, BPR, and hinge losses are a good fit for
implicit feedback models trained through negative sampling.

The regression and Poisson losses are used for explicit feedback
models.
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from torch_utils import assert_no_grad


def single_pointwise_square_loss(positive_predictions, mask=None, average=False):
    """
    square loss: (y - y_hat)^2

    This is also known as one-class prediction (i.e. no use negative sample at all).
    Learned from Neural Collaborative Filtering (Section 3.1.1)

    ref: https://dl.acm.org/citation.cfm?id=3052569
    Parameters
    ----------
    positive_predictions: :class:`torch.Tensor`
        shape (batch_size, )
    mask: :class:`numpy.ndarray`
        shape (batch_size, )
    average

    Returns
    -------

    """
    positive_labels = np.ones(positive_predictions.size()).flatten()
    is_cuda = positive_predictions.is_cuda
    if is_cuda:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
    else:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor

    positive_loss = (positive_predictions - positive_labels) * (positive_predictions - positive_labels)
    loss = positive_loss
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()
    if average:
        return loss.mean()
    else:
        return loss.sum()


def single_pointwise_bceloss(positive_predictions, mask=None, average=False):
    """
    This is cross entropy without negative sampling. Similar to classification problem where
    we only want to predict a score of a data instance in our dataset.

    This is also known as one-class prediction (i.e. no use negative sample at all).
    Learned from Neural Collaborative Filtering (Section 3.1.1)

    ref: https://dl.acm.org/citation.cfm?id=3052569
    Parameters
    ----------
    positive_predictions: :class:`torch.Tensor`
        shape (batch_size, )
    mask
    average

    Returns
    -------

    """
    positive_labels = np.ones(positive_predictions.size()).flatten()
    is_cuda = positive_predictions.is_cuda
    if is_cuda:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
    else:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor
    positive_predictions = F.sigmoid(positive_predictions)
    positive_loss = F.binary_cross_entropy(positive_predictions, positive_labels)
    loss = positive_loss
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()
    if average:
        return loss.mean()
    else:
        return loss.sum()


def pointwise_bceloss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    This is cross entropy loss. The difference is that for every positive instance, we also negative sample
    a negative instance to compute the loss.

    Learned from Neural Collaborative Filtering (Section 3.1.1)

    ref: https://dl.acm.org/citation.cfm?id=3052569

    Parameters
    ----------
    positive_predictions: :class:`torch.Tensor`
        shape (batch_size, )
    negative_predictions: :class:`torch.Tensor`
        shape (batch_size, )
    mask
    average

    Returns
    -------

    """
    positive_labels = np.ones(positive_predictions.size()).flatten()
    negative_labels = np.zeros(negative_predictions.size()).flatten()

    is_cuda = positive_predictions.is_cuda
    if is_cuda:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
        negative_labels = Variable(torch.from_numpy(negative_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
    else:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor
        negative_labels = Variable(torch.from_numpy(negative_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor

    positive_predictions = F.sigmoid(positive_predictions)
    negative_predictions = F.sigmoid(negative_predictions)

    positive_loss = F.binary_cross_entropy(positive_predictions, positive_labels)
    negative_loss = F.binary_cross_entropy(negative_predictions, negative_labels)

    loss = positive_loss + negative_loss

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()

def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    """
    Logistic loss function.

    Parameters
    ----------

    positive_predictions: :class:`torch.Tensor`
        Tensor containing predictions for known positive items.
    negative_predictions: :class:`torch.Tensor`
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    positives_loss = (1.0 - F.sigmoid(positive_predictions))
    negatives_loss = F.sigmoid(negative_predictions)

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def bpr_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """
    # old code of Spotlight
    # loss = (1.0 - F.sigmoid(positive_predictions -  negative_predictions))

    # ben: checked
    loss = - torch.log(F.sigmoid(positive_predictions - negative_predictions))

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()



def hinge_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Hinge pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    # checked, usually we need to use a threshold as soft-margin (but this function does not have it)
    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()


def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the
    ranking implicit in the pattern of user interactions.

    Approximates the idea of weighted approximate-rank pairwise loss
    introduced in [2]_

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """
    # checked
    highest_negative_predictions, _ = torch.max(negative_predictions, 0)

    return hinge_loss(positive_predictions, highest_negative_predictions.squeeze(), mask=mask)


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return ((observed_ratings - predicted_ratings) ** 2).mean()


def poisson_loss(observed_ratings, predicted_ratings):
    """
    Poisson loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return (predicted_ratings - observed_ratings * torch.log(predicted_ratings)).mean()


def logistic_loss(observed_ratings, predicted_ratings):
    """
    Logistic loss for explicit data.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings which
        should be +1 or -1 for this loss function.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    # Convert target classes from (-1, 1) to (0, 1)
    observed_ratings = torch.clamp(observed_ratings, 0, 1)

    return F.binary_cross_entropy_with_logits(predicted_ratings,
                                              observed_ratings,
                                              size_average=True)
