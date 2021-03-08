from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GraphAttentionLayer
from evaluation import eva


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(DAEGC, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred,z


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class Self_DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(Self_DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        #pre_daegc
        self.pre_daegc = DAEGC(num_features, hidden_size, embedding_size, alpha)
        self.pre_daegc.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z = self.pre_daegc(x, adj, M)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, z, q


def daegc(dataset):
    model = Self_DAEGC(num_features=args.input_dim, hidden_size=args.hidden1_dim,
                  embedding_size=args.hidden2_dim, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    with torch.no_grad():
        _, z = model.pre_daegc(data, adj, M)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    #y_pred_last = y_pred       #kmeans.cluster_centers_  从z中获得初始聚类中心
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(30):
        model.train()
        if epoch % args.update_interval == 0:      #[1,3,5]
            # update_interval
            A_pred, z, tmp_q = model(data, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            # eva(y, res3, str(epoch) + 'P')

        A_pred, z, q = model(data, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cite')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden1_dim', default=256, type=int)
    parser.add_argument('--hidden2_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.name)

    if args.name == 'cite':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
        args.input_dim = 3703
        args.pretrain_path = 'predaegc_cite.pkl'


    print(args)
    daegc(dataset)