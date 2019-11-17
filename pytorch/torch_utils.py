import numpy as np

import torch


def flatten(x):
    '''
    flatten high dimensional tensor x into an array
    :param x: shape (B, D1, D2, ...)
    :return: 1 dimensional tensor
    '''
    dims = x.size()[1:]  # remove the first dimension as it is batch dimension
    num_features = 1
    for s in dims: num_features *= s
    return x.contiguous().view(-1, num_features)


def gpu(tensor, gpu=False):
    if gpu: return tensor.cuda()
    else: return tensor


def cpu(tensor):

    if tensor.is_cuda: return tensor.cpu()
    else: return tensor


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )

def numpy2tensor(x, dtype):
    # torch.tensor(torch.from_numpy(var), dtype = torch.int)
    return torch.tensor(torch.from_numpy(x), dtype = dtype)

def tensor2numpy(x):
    # return x.numpy()
    return cpu(x).numpy()

def set_seed(seed, cuda=False):

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


def _predict_process_ids(user_ids, item_ids, num_items, use_cuda):
    """

    Parameters
    ----------
    user_ids
    item_ids
    num_items
    use_cuda

    Returns
    -------

    """
    if item_ids is None:
        item_ids = np.arange(num_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if item_ids.size()[0] != user_ids.size(0):
        user_ids = user_ids.expand(item_ids.size())

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()