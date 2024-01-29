import torch
import torch.nn as nn


class DendroPRS(nn.Module):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False, init_roots=True):
        """
        :param device: specify cpu or cuda device
        :param root_weights: empty np-array of dimension n_features # todo: make this a dimension and instantiate tensor
        :param path_mat: index array of dimension (n_edges, n_nodes), nodes corresponding to population tree
        Each column i contains 1 at position j if edge j is in the path from root to node i
        :param delta_mat: empty np-array of dimension (n_features, n_edges) # todo: make this a dimension and instantiate tensor
        :param p: norm for use in calculating regularization penalty for deltas
        :param init_deltas: if True, initialize deltas to small random values
        :param init_roots: if True, initialize root weights to small random values
        """
        super(DendroPRS, self).__init__()
        self.device = device
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)
        self.p = p
        self.root_weights = nn.Parameter(torch.tensor(root_weights, device=device, dtype=torch.double, requires_grad=True))
        if init_roots:
            torch.nn.init.normal_(self.root_weights, mean=0.0, std=0.01)

        self.delta_mat = nn.Parameter(torch.tensor(delta_mat, device=device, dtype=torch.double, requires_grad=True))
        if init_deltas:
            torch.nn.init.normal_(self.delta_mat, mean=0.0, std=0.01)

    def group_lasso(self):
        stacked_tensor = torch.cat((torch.unsqueeze(self.root_weights, dim=1), self.delta_mat), dim=1)
        return torch.sum(torch.norm(stacked_tensor, p=2, dim=1))

    def delta_loss(self, idx, rows=None):  # messy looking, but avoids unnecessary slicing or instantiating rows list
        if idx is not None:
            edges = torch.max(self.path_mat[:, idx], dim=1)
            if rows is None:
                mat_slice = self.delta_mat.T[torch.nonzero(edges.values == 1.0).reshape(-1)]
            else:
                mat_slice = self.delta_mat[rows].T[torch.nonzero(edges.values == 1.0).reshape(-1)]
            return torch.norm(mat_slice, p=self.p)

        if rows is None:
            return torch.norm(self.delta_mat, p=self.p)
        else:
            return torch.norm(self.delta_mat[rows], p=self.p)

    # node_idx identifies the paths relevant to all samples in x, in the same order
    def forward(self, x, node_idx):
        effective_weights = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        # assumes linear weights with a bias feature included
        return torch.sum((x * effective_weights), dim=1)
