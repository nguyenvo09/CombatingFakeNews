import layers
import torch_utils
from torch import nn
import torch


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, pretrained_user_embeddings = None, pretrained_item_embeddings = None):
        super(MF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = embedding_dim

        if pretrained_user_embeddings is not None:
            self.user_embeddings = pretrained_user_embeddings
        else:
            self.user_embeddings = layers.ScaledEmbedding(n_users, embedding_dim)

        if pretrained_item_embeddings is not None:
            self.item_embeddings = pretrained_item_embeddings
        else:
            self.item_embeddings = layers.ScaledEmbedding(n_items, embedding_dim)

        self.user_bias = layers.ZeroEmbedding(n_users, 1)
        self.item_bias = layers.ZeroEmbedding(n_items, 1)

    def forward(self, uids, iids):
        user_embeds = self.user_embeddings(uids) #first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = torch_utils.flatten(user_embeds)
        item_embeds = torch_utils.flatten(item_embeds)

        user_bias = self.user_bias(uids)
        item_bias = self.item_bias(iids)

        user_bias = torch_utils.flatten(user_bias) #bias has size batch_size * 1
        item_bias = torch_utils.flatten(item_bias) #bias has size batch_size * 1

        dot_product = (user_embeds * item_embeds).sum(1) #first dimension is batch_size, return dimension (batch_size)
        # dot_product = torch.mul(user_embeds, item_embeds).sum(1)  # first dimension is batch_size

        return dot_product + user_bias.squeeze() + item_bias.squeeze()


class GAU(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(GAU, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = embedding_dim

        self.user_embeddings = layers.ScaledEmbedding(n_users, embedding_dim)
        self.item_embeddings = layers.ScaledEmbedding(n_items, embedding_dim)

        # addtional parameters in GAU model
        self.user_additional = layers.ScaledEmbedding(n_users, embedding_dim)
        self.item_additional = layers.ScaledEmbedding(n_items, embedding_dim)

        self.user_bias = layers.ZeroEmbedding(n_users, 1)
        self.item_bias = layers.ZeroEmbedding(n_items, 1)

    def forward(self, uids, iids, network = None):
        """

        :param uids: shape (B, )
        :param iids: shape (B, )
        :param network: tuple where first element is a tensor array of shape (X, ), shape (X, total-number-of-user) where X <= B
        :return:
        """
        user_embeds = self.user_embeddings(uids) #first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = torch_utils.flatten(user_embeds)
        item_embeds = torch_utils.flatten(item_embeds)

        user_bias = self.user_bias(uids)
        item_bias = self.item_bias(iids)

        user_bias = torch_utils.flatten(user_bias) #bias has size batch_size * 1
        item_bias = torch_utils.flatten(item_bias) #bias has size batch_size * 1
        loss = (user_embeds * item_embeds).sum(1) + \
               user_bias.squeeze() + item_bias.squeeze() # first dimension is batch_size, return dimension (batch_size)

        return loss

    def network_loss(self, network):
        target, binary_weights = network
        target_embeds = self.user_embeddings(target)  # shape (X, D)
        W = torch.mm(target_embeds, torch.transpose(self.user_embeddings.weight, 0, 1))  # (X, D) x (n_users x D)
        network_loss = ((W - binary_weights) ** 2).sum()
        return network_loss

    def user_user_sppmi_loss(self, sppmi):
        target, weights = sppmi
        target_embeds = self.user_embeddings(target)  # shape (X, D)
        W = torch.mm(target_embeds, torch.transpose(self.user_additional.weight, 0, 1))  # (X, D) x (n_users x D)
        mask = weights > 0
        A = mask.type(torch.float) * (W - weights)
        sppmi_loss = (A ** 2).sum()
        return sppmi_loss

    def item_item_sppmi_loss(self, sppmi):
        target, weights = sppmi
        target_embeds = self.item_embeddings(target)  # shape (X, D)
        W = torch.mm(target_embeds, torch.transpose(self.item_additional.weight, 0, 1))  # (X, D) x (n_users x D)
        mask = weights > 0
        A = mask.type(torch.float) * (W - weights)
        sppmi_loss = (A ** 2).sum()
        return sppmi_loss

    def user_user_sim_loss(self, sim):
        target, weights = sim
        N, s, D = self.n_users, len(target), self.n_factors
        target_embeds = self.user_embeddings(target)  # shape (X, D)
        B = target_embeds.repeat(N, 1).view(N, s, D).transpose(0, 1)
        A = B - self.user_embeddings.weight
        A = (A ** 2).sum(dim = -1)
        assert A.shape == (s, N)
        assert weights.shape == (s, N)
        loss = (A * weights).sum()
        # print("here, ", loss )
        return loss

    def item_item_sim_loss(self, sim):
        target, weights = sim
        N, s, D = self.n_items, len(target), self.n_factors
        target_embeds = self.item_embeddings(target)  # shape (X, D)
        B = target_embeds.repeat(N, 1).view(N, s, D).transpose(0, 1)
        A = B - self.item_embeddings.weight
        # print(A)
        A = (A ** 2).sum(dim = -1)
        assert A.shape == (s, N)
        assert weights.shape == (s, N)
        # print(weights)
        loss = (A * weights).sum()
        # print("item-item-sim, ", loss )
        return loss

    def predict(self, uids, iids):
        user_embeds = self.user_embeddings(uids)  # first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = torch_utils.flatten(user_embeds)
        item_embeds = torch_utils.flatten(item_embeds)

        dot_products = (user_embeds * item_embeds).sum(1)

        return dot_products

