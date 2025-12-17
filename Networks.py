import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import torch.optim as optim


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def knn_adjacency(features, k=500, metric='cosine'):

    num_nodes = features.size(0)
    if metric == 'cosine':
        similarity = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    elif metric == 'euclidean':
        feat_expand_1 = features.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        feat_expand_2 = features.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        distance = torch.norm(feat_expand_1 - feat_expand_2, p=2, dim=2)
        similarity = 1 / (1 + distance)
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    _, indices = torch.topk(similarity, k=k, dim=1, largest=True, sorted=False)
    adjacency = torch.zeros(num_nodes, num_nodes, device=features.device)
    adjacency.scatter_(1, indices, 1)
    adjacency = torch.max(adjacency, adjacency.t())

    return adjacency


def total_variation_smooth_loss(features, adjacency):
    edge_indices = adjacency.nonzero(as_tuple=False)
    diff = features[edge_indices[:, 0]] - features[edge_indices[:, 1]]
    loss = torch.sum(torch.norm(diff, p=1, dim=1))
    return loss


# ==============================================================================
# NETWORK MODULES
# ==============================================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_rate):
        super(GCNEncoder, self).__init__()
        self.gcn_layer = GCNConv(in_channels, hidden_channels)
        self.drop_layer = nn.Dropout(drop_rate)

    def forward(self, feats, edges):
        h = F.relu(self.gcn_layer(feats, edges))
        h = self.drop_layer(h)
        return h


class GCNDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, drop_rate):
        super(GCNDecoder, self).__init__()
        self.gcn_layer = GCNConv(hidden_channels, out_channels)
        self.drop_layer = nn.Dropout(drop_rate)

    def forward(self, h, edges):
        h = self.drop_layer(h)
        out = F.relu(self.gcn_layer(h, edges))
        return out


class DualStreamGAE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop_prob):
        super(DualStreamGAE, self).__init__()
        self.shared_enc = GCNEncoder(in_dim, hid_dim, drop_prob)
        self.dec_stream_pre = GCNDecoder(hid_dim, out_dim, drop_prob)
        self.dec_stream_post = GCNDecoder(hid_dim, out_dim, drop_prob)

    def forward(self, features, edge_idx, mode):
        latent = self.shared_enc(features, edge_idx)

        if mode == 'pre':
            reconstructed = self.dec_stream_pre(latent, edge_idx)
        elif mode == 'post':
            reconstructed = self.dec_stream_post(latent, edge_idx)
        else:
            raise ValueError("Mode must be either 'pre' or 'post'")

        return reconstructed


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train_model(feat_dim, total_sp_count, feats_pre, feats_post,
                adj_pre, adj_post, lap_pre, lap_post, n_seg, config):

    dropout = config['dropout']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config['epoch']
    hidden_dim = config['hidden_dim']
    knn_ratio = config['knn_ratio']

    cross_lambda = config['cross_lambda']
    cycle_lambda = config['cycle_lambda']
    structure_lambda = config['structure_lambda']
    delt_lambda = config['delt_lambda']
    align_lambda = config['align_lambda']
    smooth_lambda = config['smooth_lambda']

    k_neighbors = int(knn_ratio * n_seg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    delta_pre = torch.ones(total_sp_count, feat_dim, requires_grad=True, device=device)
    delta_post = torch.ones(total_sp_count, feat_dim, requires_grad=True, device=device)

    t_feats_pre = torch.tensor(feats_pre, dtype=torch.float32, device=device)
    t_feats_post = torch.tensor(feats_post, dtype=torch.float32, device=device)

    adj_pre = adj_pre.to(device)
    adj_post = adj_post.to(device)
    lap_pre = lap_pre.to(device)
    lap_post = lap_post.to(device)

    model = DualStreamGAE(in_dim=feat_dim, hid_dim=hidden_dim,
                          out_dim=feat_dim, drop_prob=dropout).to(device)
    model.train()

    optimizer = optim.Adam([delta_pre, delta_post] + list(model.parameters()),
                           lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        rec_pre = model(t_feats_pre + delta_pre, adj_post, mode='pre')
        rec_post = model(t_feats_post + delta_post, adj_pre, mode='post')
        loss_rec_pre = torch.norm(t_feats_pre - rec_post, p='fro') ** 2
        loss_rec_post = torch.norm(t_feats_post - rec_pre, p='fro') ** 2
        reconstruct_loss = (loss_rec_pre + loss_rec_post) / total_sp_count

        cycle_pre = model(rec_post + delta_pre, adj_post, mode='pre')
        cycle_post = model(rec_pre + delta_post, adj_pre, mode='post')

        loss_cycle_pre = torch.norm(t_feats_pre - cycle_pre, p='fro') ** 2
        loss_cycle_post = torch.norm(t_feats_post - cycle_post, p='fro') ** 2
        cycle_loss = (loss_cycle_pre + loss_cycle_post) / total_sp_count

        struct_loss_pre = 2 * torch.trace((t_feats_pre + delta_pre).t() @ lap_post @ (t_feats_pre + delta_pre))
        struct_loss_post = 2 * torch.trace((t_feats_post + delta_post).t() @ lap_pre @ (t_feats_post + delta_post))
        structure_loss = (struct_loss_pre + struct_loss_post) / total_sp_count

        d_loss = (torch.norm(delta_pre, p='fro') ** 2 + torch.norm(delta_post, p='fro') ** 2) / total_sp_count

        delta_pre_sq = torch.sum((delta_pre) ** 2, dim=1)
        delta_post_sq = torch.sum((delta_post) ** 2, dim=1)
        align_loss = torch.exp(-delta_pre_sq * delta_post_sq).sum() / n_seg

        lap_delta_pre = knn_adjacency(delta_pre, k=k_neighbors, metric='cosine')
        lap_delta_post = knn_adjacency(delta_post, k=k_neighbors, metric='cosine')
        smooth_loss_pre = total_variation_smooth_loss(delta_pre, lap_delta_pre) / n_seg
        smooth_loss_post = total_variation_smooth_loss(delta_post, lap_delta_post) / n_seg
        smooth_loss = (smooth_loss_pre + smooth_loss_post) / k_neighbors

        loss = (
                cross_lambda * reconstruct_loss
                + cycle_lambda * cycle_loss
                + structure_lambda * structure_loss
                + delt_lambda * d_loss
                + align_lambda * align_loss
                + smooth_lambda * smooth_loss
        )

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    out_cycle_pre = cycle_pre.cpu().detach().numpy()
    out_rec_pre = rec_pre.cpu().detach().numpy()
    out_delta_pre = delta_pre.cpu().detach().numpy()

    out_cycle_post = cycle_post.cpu().detach().numpy()
    out_rec_post = rec_post.cpu().detach().numpy()
    out_delta_post = delta_post.cpu().detach().numpy()

    return out_cycle_pre, out_rec_pre, out_delta_pre, out_cycle_post, out_rec_post, out_delta_post