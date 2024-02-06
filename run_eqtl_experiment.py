import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr
from models import DendroPRS
from utils import create_directory, split_indices
from model_utils import build_parent_path_mat, newick_to_adjacency_matrix, IndicesDataset

USE_CUDA = False
L1_ROOTS = True
GROUP_LASSO = True
print('Using CUDA: ' + str(USE_CUDA))
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

# superpops order corresponds to cluster assignments - hard-coded for 1000 Genomes super-populations with constant branch lengths
tree_string = '((((AMR:1.00,EAS:1.00):1.00,SAS:1.00):1.00,EUR:1.00):1.00,AFR:1.00);'
pops = ['AFR', 'EUR', 'SAS', 'EAS', 'AMR']

# parameter settings, pointers to data files
cluster_file = 'data/phase3_clusters.csv'
bias_in = True


def get_pheno_key(phenotype, pheno_keys):
    # addressing a slight key format change
    for key in pheno_keys:
        if key.split('.')[0] == phenotype:
            return key


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotypes-file', type=str, default='example_data/ENSG00000175854_genotypes')
    parser.add_argument('--phenotypes-file', type=str, default='example_data/example_phenotypes.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crossval-splits', type=int, default=5)
    parser.add_argument('--early-stopping', type=int, default=10, help='Number of epochs without improvement before early stopping')
    parser.add_argument('--validation-interval', type=int, default=1)
    parser.add_argument('--dpf', type=float, default=0.1, help='scaling factor applied to delta term in the loss')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--group-lasso', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='eqtl_experiment')
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    # directory removal warning!
    base_output_dir = os.path.join(os.path.abspath('.'), 'experiment_output', args.output_dir)
    create_directory(base_output_dir, remove_curr=False)

    """
    As we do not have phenotype data for all samples, we select the relevant portions of the 
    genome and cluster map files, based around the ID indices in the cluster map file
    """
    cluster_map = pd.read_csv(cluster_file, sep='\t', header=None).to_numpy()

    pheno_df = pd.read_csv(args.phenotypes_file)

    with open(args.genotypes_file, 'rb') as f:
        genome = pickle.load(f)[0]
    f.close()

    # inferring phenotype from the genotype name
    phenotype = args.genotypes_file.split('/')[-1].split('_')[0]

    present_sample_idx = list()
    pheno_col = pheno_df[phenotype].to_numpy()
    present_sample_ID = list()
    id_col = pheno_df['Reporter Identifier'].to_numpy()
    for idx, sample_id in enumerate(cluster_map[:, 0]):
        if sample_id in id_col:
            present_sample_idx.append(idx)
            present_sample_ID.append(sample_id)

    # uncomment to save the present indices and IDs
    # pd.DataFrame(data=present_sample_ID, columns=['Present IDs']).to_csv('present_samples_ID.csv', index=False)

    genome = genome[present_sample_idx]
    cluster_map = cluster_map[present_sample_idx][:, 2]
    # todo: <- done, could factor out this code
    y = list()
    for sample_id in present_sample_ID:
        y.append(float(pheno_df[pheno_df['Reporter Identifier'] == sample_id][phenotype].values[0]))
    y = np.asarray(y)
    # log-transforming the gene expression values
    y = np.log(np.clip(np.asarray(y, dtype=float), 0.00000001, None))

    num_samples = genome.shape[0]
    assert y.shape[0] == num_samples
    # adding a bias feature to the genome data
    if bias_in:
        genome = np.hstack((genome, np.expand_dims(np.asarray([1.0 for _ in range(num_samples)]), axis=0).transpose()))

    pp_ordered_nodes, parent_child_mat = newick_to_adjacency_matrix(tree_string, pops)
    parent_path = build_parent_path_mat(parent_child_mat)

    num_edges = parent_path.shape[0]
    num_features = genome.shape[1]
    edge_matrix = np.zeros(shape=(num_features, num_edges), dtype=np.float)

    path_dict = dict()
    pp_ordered_nodes = [node.name for node in pp_ordered_nodes]
    for sample_idx, cluster_idx in enumerate(cluster_map):
        path_dict[sample_idx] = pp_ordered_nodes.index(pops[cluster_idx])

    X_tensor = torch.tensor(genome, dtype=torch.double, device=device)
    y_tensor = torch.tensor(y, dtype=torch.double, device=device)

    # creating a dict to store the metrics after training
    metrics_dict = {
        'dendro_mse': list(),
        'dendro_corr': list(),
        'dendro_idx': list(),
        'dendro_preds': list(),
        'dendro_test_corr': list(),
        'dendro_test_preds': list(),
    }

    for seed in range(args.crossval_splits):
        crossval_idx, test_idx = split_indices(list(range(num_samples)),
                                               seed=seed)
        # uncomment line below to save the train/test split indices
        # pd.DataFrame(data=np.asarray(test_idx), columns=['Test Indices']).to_csv(
        #     '{}/test_indices_rep{}.csv'.format(base_output_dir, seed), index=False)

        test_idx_tensor = torch.tensor(test_idx)
        test_path_tensor = torch.tensor([path_dict[idx] for idx in test_idx])

        root_weights = np.zeros(num_features)
        train_idx, valid_idx = split_indices(crossval_idx, seed=seed)

        # tuples of idx in X array and corresponding path idx
        # creating idx dataset objects for batching
        train_set = IndicesDataset(np.asarray([(idx, path_dict[idx]) for idx in train_idx], dtype=np.int64))
        valid_set = IndicesDataset(np.asarray([(idx, path_dict[idx]) for idx in valid_idx], dtype=np.int64))

        # Parameters for shuffle batch
        params = {'batch_size': 32,
                  'shuffle': True,
                  'num_workers': 0}

        train_batch_gen = torch.utils.data.DataLoader(train_set, **params)
        valid_batch_gen = torch.utils.data.DataLoader(valid_set, **params)

        dendronet = DendroPRS(device, root_weights, parent_path, edge_matrix, init_deltas=False,
                                       p=args.p)

        loss_function = nn.MSELoss()
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(dendronet.parameters(), lr=args.lr)

        # creating a tuple tracking best validation result + epoch for early stopping
        best_result = (np.inf, 0)
        best_avg_loss, best_corr, best_idx, best_preds, test_corr, test_preds = None, None, None, None, None, None

        # training loop for dendronet
        for epoch in range(args.epochs):
            print('Train epoch ' + str(epoch))
            running_loss = 0.0
            running_mse_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                y_hat = dendronet.forward(X_tensor[idx_batch[:, 0]], idx_batch[:, 1])
                delta_loss = dendronet.delta_loss(idx=idx_batch[:, 1])
                train_loss = loss_function(y_hat, y_tensor[idx_batch[:, 0]])
                running_mse_loss += float(train_loss.detach().cpu().numpy())
                loss = train_loss + (delta_loss * args.dpf)
                if L1_ROOTS:
                    l1_root = args.l1 * torch.norm(dendronet.root_weights, 1)
                    loss = loss + l1_root
                if GROUP_LASSO:
                    group_loss = dendronet.group_lasso() * args.group_lasso
                    loss = loss + group_loss
                loss.backward(retain_graph=True)
                running_loss += float(loss.detach().cpu().numpy())
                optimizer.step()
            print('Avg MSE loss ', running_mse_loss / (step + 1))
            if epoch % args.validation_interval == 0 or epoch == args.epochs-1:
                running_valid_loss = 0.0
                running_valid_preds = list()
                running_valid_labels = list()
                tmp_batch_tracker = list()
                print('Valid epoch ' + str(epoch))
                with torch.no_grad():
                    for step, idx_batch in enumerate(tqdm(valid_batch_gen)):
                        tmp_batch_tracker.extend(list(idx_batch.detach().cpu().numpy()[:, 0]))
                        targets = y_tensor[idx_batch[:, 0]]
                        valid_y_hat = dendronet.forward(X_tensor[idx_batch[:, 0]], idx_batch[:, 1])
                        valid_loss = loss_function(valid_y_hat, targets)
                        running_valid_preds.extend(np.asarray(valid_y_hat.detach().cpu().numpy(), dtype=float))
                        if type(targets) == np.float64:
                            running_valid_labels.append(targets)
                        else:
                            running_valid_labels.extend(targets)
                        running_valid_loss += float(valid_loss.detach().cpu().numpy())
                avg_loss = loss_function(torch.tensor(running_valid_preds, dtype=torch.double),
                                         torch.tensor(running_valid_labels, dtype=torch.double))
                corr = pearsonr(np.asarray(running_valid_preds),
                                [float(i.detach().cpu().numpy()) for i in running_valid_labels])[0]
                print('Average valid loss ', avg_loss)
                print('Correlation: ', corr)
                if running_valid_loss < best_result[0]:
                    best_result = (running_valid_loss, epoch)
                    best_avg_loss = float(avg_loss.detach().cpu().numpy())
                    best_corr = corr
                    best_idx = tmp_batch_tracker
                    best_preds = running_valid_preds
                    """
                    running the test set, could optimize to be batchable
                    """
                    with torch.no_grad():
                        test_targets = y_tensor[test_idx]
                        test_preds = dendronet.forward(X_tensor[test_idx_tensor], test_path_tensor)
                        test_corr = pearsonr(test_targets,
                                [float(i.detach().cpu().numpy()) for i in test_preds])[0]

            # done training, store results
            if epoch == args.epochs - 1 or epoch - best_result[1] >= args.early_stopping:
                metrics_dict['dendro_mse'].append(best_avg_loss)
                metrics_dict['dendro_corr'].append(best_corr)
                metrics_dict['dendro_idx'].extend(best_idx)
                metrics_dict['dendro_preds'].extend(best_preds)
                metrics_dict['dendro_test_corr'].append(test_corr)
                metrics_dict['dendro_test_preds'].extend(test_preds.detach().cpu().numpy())
                torch.save(dendronet.state_dict(), '{}_dendronet_model'.format(base_output_dir))
                print('Ending training at epoch', epoch)
                break

    metrics_dict['dendro_avg'] = str(np.mean(metrics_dict['dendro_mse'])) + ' +/- ' + \
                                 str(np.std(metrics_dict['dendro_mse']))
    metrics_dict['dendro_corr_avg'] = str(np.mean(metrics_dict['dendro_corr'])) + ' +/- ' + \
                                 str(np.std(metrics_dict['dendro_corr']))

    with open(os.path.join(base_output_dir, "metric_results"), 'wb') as f:
        pickle.dump([metrics_dict], f)