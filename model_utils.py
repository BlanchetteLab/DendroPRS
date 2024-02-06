"""
Helper functions and classes for DendroPRS
"""
import io
import torch
import numpy as np
from Bio import Phylo


# adapted from https://biopython.org/wiki/Phylo_cookbook
def newick_to_adjacency_matrix(tree_string, pops_list=None):
    """Create an adjacency matrix (NumPy array) from clades/branches in tree.

    Also returns a list of all clades in tree ("allclades"), where the position
    of each clade in the list corresponds to a row and column of the numpy
    array: a cell (i,j) in the array is 1 if there is a branch from allclades[i]
    to allclades[j], otherwise 0.

    Returns a tuple of (allclades, adjacency_matrix) where allclades is a list
    of clades and adjacency_matrix is a NumPy 2D array.
    """
    tree = Phylo.read(io.StringIO(tree_string), "newick")

    allclades = list(tree.find_clades(order="level"))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    adjmat = np.zeros((len(allclades), len(allclades)), dtype=int)
    for parent in tree.find_clades(terminal=False, order="level"):
        for child in parent.clades:
            adjmat[lookup[parent], lookup[child]] = 1
    return (allclades, adjmat)


"""
Takes in list of relevant indices, 
for use with the dataloader class for batching when full dataset is an array in memory
"""
class IndicesDataset(torch.utils.data.Dataset):
  def __init__(self, sample_indices):
        self.sample_indices = sample_indices

  def __len__(self):
        return len(self.sample_indices)

  def __getitem__(self, index):
        'Generates an index for one sample of data'
        # Select sample
        sample_idx = self.sample_indices[index]

        return sample_idx

# could be easily altered to work with weighted paths as well
def build_parent_path_mat(parent_child_mat, num_edges=None):
    """
    param parent_child_mat: np binary array, rows-> parent cols-> child, first row must be root
    -note that the parent-child mat must be topologically ordered
    return: parent_path matrix: np array, rows->edges cols->nodes
    """
    num_nodes = parent_child_mat.shape[0]
    # if num_edges is not passed in, counting the number of edges above the diagonal
    if num_edges is None:
        num_edges = np.sum(np.triu(parent_child_mat, 1))

    parent_path = np.zeros(shape=(num_nodes, num_edges), dtype=np.float32)
    edge_index = 0

    for node_index in range(1, num_nodes):  # skipping the root node, which we know has an empty parent path
        # edge to parent becomes a new edge
        parent_path[node_index, edge_index] = 1.0
        edge_index += 1
        # find the parent node via edge mat, add parent path values
        parent_node_idx = np.where(parent_child_mat[:, node_index] == 1)[0]
        prev_pp_idx = np.where(parent_path[parent_node_idx] == 1.0)
        for idx in prev_pp_idx:
            parent_path[node_index, idx] = 1.0

    # taking the transpose so that every column holds all the relevant edges for a node
    return np.transpose(parent_path)