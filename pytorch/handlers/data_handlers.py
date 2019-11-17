import numpy as np
import collections
from tqdm import tqdm
import pandas as pd
from scipy import sparse
import torch_utils


# reuse from neural collaborative filtering
def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


# reuse from neural collaborative filtering
def load_negative_file(ratings, filename):
    """
    We need to make sure the consistency of negative samples per interaction.
    Parameters
    ----------
    ratings: :class:`list[tuple]` list of ratings
    filename: :class:`string` filename of negative
    samples with format (u,i) neg1 neg2 neg3
    Returns
    -------

    """
    negativeList = []
    cnt = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            point = arr[0][1:-1]
            a, b = point.split(",")
            point = [int(a), int(b)]
            assert ratings[cnt] == point
            cnt += 1
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    assert len(ratings) == len(negativeList)
    return negativeList


def load_network(adj_network):
    """
    Loading adjacent network information.

    :param adj_network:
    :return: an adjacent network for every vertex (every user)
    """
    fin = open(adj_network, "r")
    graph = collections.defaultdict(list)
    for line in tqdm(fin):
        line = line.replace("\n", "")
        args = line.split("\t")
        node = args[0].replace("(", "")
        node = node.replace(")", "")
        node = int(node)
        assert node not in graph
        graph[node] = list(map(float, args[1:]))
    return graph


def computeSPPMI_matrix(P, shifted_k: int, check_sym = True):
    """
    Computing SPPMI matrix. Expected is a symmetric matrix. Diagonal is expected to be all zeros.
    :param P: A symmetric square matrix where each element is the frequency of pair (i, j).
    :param shifted_k: `int` number of negative samples
    :param check_sym: `bool` checking symmetric or not.
    """
    a, b = P.shape
    # print P.shape, type(P)
    P = np.asarray(P)
    # print 'Shape of P: ', P.shape
    assert a == b
    D = np.sum(P)
    cols = np.sum(P, axis = 0)  # sum the columns (b,)
    rows = np.sum(P, axis = 1, keepdims=True)  # sum the row (a,1)
    P = P / (cols + 1e-10)
    P = P / (rows + 1e-10)
    P = P * D
    PMI = np.log(P + 1e-10)
    S = PMI - np.log(shifted_k)
    mask = S > 0
    SPPMI = np.multiply(S, mask)
    assert np.min(SPPMI) == 0
    if check_sym: np.fill_diagonal(SPPMI, 0)  # in-place operation
    assert torch_utils.check_symmetric(SPPMI) == True
    return SPPMI


def load_data_item_item_for_sppmi(csvfile, no_items: int, check_sym = True):
    '''
    We need to return co-occurrence matrix of item-item and guadians-guardians
    :param csvfile:
    :param n_guardians:
    :param n_urls:
    :return:
    '''
    tp = pd.read_csv(csvfile)
    rows, cols, freqs = np.array(tp['item1']), np.array(tp['item2']), np.array(tp['freq'])
    assert freqs.shape == rows.shape and rows.shape == cols.shape

    weights = sparse.csr_matrix((np.array(tp['freq']), (rows, cols)), dtype=np.int16, shape=(no_items, no_items))
    if check_sym: assert torch_utils.check_symmetric(weights.todense()) == True
    return weights.todense()


def load_sim(csvfile, N):
    '''
    We need to return co-occurrence matrix of item-item and guadians-guardians (keke)
    :param csvfile:
    :param n_guardians:
    :param n_urls:
    :return:
    '''
    tp = pd.read_csv(csvfile, header = None, sep = ",")
    rows, cols, freqs = np.array(tp[0]), np.array(tp[1]), np.array(tp[2])
    assert freqs.shape == rows.shape and rows.shape == cols.shape

    weights = sparse.csr_matrix((np.array(tp[2]), (rows, cols)), dtype=np.float32, shape=(N, N))

    return weights.todense()


def load_data_user_user_for_sppmi(csvfile, no_guardians):
    '''
    We need to return co-occurrence matrix of item-item and guadians-guardians
    :param csvfile:
    :param n_guardians:
    :param n_urls:
    :return:
    '''
    tp = pd.read_csv(csvfile)
    rows, cols, freqs = np.array(tp['guardian1']), np.array(tp['guardian2']), np.array(tp['freq'])
    assert freqs.shape == rows.shape and rows.shape == cols.shape

    weights = sparse.csr_matrix((np.array(tp['freq']), (rows, cols)), dtype=np.int16, shape=(no_guardians, no_guardians))
    return weights.todense()


if __name__ == '__main__':
    pass
