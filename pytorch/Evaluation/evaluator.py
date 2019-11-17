import math
import numpy as np
import collections


def evaluate(model, testRatings, testNegatives, K):
    """
    We could extend it to add more metrics in the future
    Parameters
    ----------
    model
    testRatings
    testNegatives
    K

    Returns
    -------

    """
    hits, ndcgs = [], []
    i = 0
    for idx in range(len(testRatings)):
        i+=1
        (hr, ndcg) = eval_one_rating(model, testRatings, testNegatives, K, idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        # if gc.DEBUG:
        #    if i >= 10: break# break # debug

    results = {}
    results["ndcg"] = np.nanmean(ndcgs)
    results["ndcg_list"] = ndcgs
    results["hits"] = np.nanmean(hits)
    results["hits_list"] = hits

    return results

    # return np.nanmean(hits), np.nanmean(ndcgs)


def eval_one_rating(model, testRatings, testNegatives, K, idx, begin_at_0=True):
    """
    No need for padding (dkm)
    Parameters
    ----------
    model
    testRatings
    testNegatives
    K
    idx
    begin_at_0

    Returns
    -------

    """
    rating = testRatings[idx]
    items = testNegatives[idx]
    uid, iid = rating[0], rating[1]
    # items.append(iid) # in-place ? Not good wrong Bac Ky haha
    items = items + [iid]  # No longer in-place
    assert len(items) == 100

    users = np.full(len(items), uid, dtype=np.int64)
    items = np.asarray(items)
    predictions = model.predict(users, items)

    # # Evaluate top rank list
    # ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    # if uid <= 1:
    #     print('--> ',predictions)
    #     print("users: ", users)
    #     print("items: ", items)
    indices = np.argsort(-predictions)[:K] # indices of items with highest scores
    ranklist = items[indices]

    hr = getHitRatio(ranklist, iid)
    ndcg = getNDCG(ranklist, iid)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    # print(ranklist, gtItem)
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2.0) / math.log(i+2.0)
    return 0
