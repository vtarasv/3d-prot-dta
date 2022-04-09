from metrics import *

import math
import pickle
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, \
    f1_score, auc, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error, max_error, r2_score
from scipy.stats import pearsonr


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class Graph:
    pass


# noinspection PyTypeChecker
class BatchGenerator:
    def __init__(self, chem_key_to_graph_dict, prot_key_to_graph_dict,
                 chem_key_to_meta_dict, prot_key_to_meta_dict, device,
                 chem_node_features_len=None, prot_node_features_len=None,
                 chem_edge_features_len=None, prot_edge_features_len=None):

        self.chem_key_to_graph_dict = chem_key_to_graph_dict
        self.prot_key_to_graph_dict = prot_key_to_graph_dict
        self.chem_key_to_meta_dict = chem_key_to_meta_dict
        self.prot_key_to_meta_dict = prot_key_to_meta_dict
        self.device = device
        self.chem_node_features_len = chem_node_features_len
        self.prot_node_features_len = prot_node_features_len
        self.chem_edge_features_len = chem_edge_features_len
        self.prot_edge_features_len = prot_edge_features_len

        self.chem_gen_edge_features = False
        self.prot_gen_edge_features = False

        self.chem_batch_dict = {}
        self.prot_batch_dict = {}
        self.y = None

    def get_batch(self, df_batch, *, chem_graph_key="", prot_graph_key="", chem_meta_key="", prot_meta_key="",
                  label_key="", chem_gen_edge_features=True, prot_gen_edge_features=True):
        self.reset()
        self.chem_gen_edge_features = chem_gen_edge_features
        self.prot_gen_edge_features = prot_gen_edge_features

        df_batch.reset_index(drop=True, inplace=True)

        self.init_arrays(df_batch, chem_graph_key, prot_graph_key)

        for ind, row in df_batch.iterrows():
            self.get_arrays(row[chem_graph_key], self.chem_key_to_graph_dict, self.chem_batch_dict, ind,
                            self.chem_gen_edge_features)
            self.get_arrays(row[prot_graph_key], self.prot_key_to_graph_dict, self.prot_batch_dict, ind,
                            self.prot_gen_edge_features)

        self.conv_arrays_to_tensors(self.chem_batch_dict, self.chem_gen_edge_features)
        self.conv_arrays_to_tensors(self.prot_batch_dict, self.prot_gen_edge_features)
        self.y = torch.from_numpy(np.array(df_batch[label_key])).type(torch.float32).view(-1, 1).to(self.device)

        batch_chem, batch_prot = Graph(), Graph()
        batch_chem.x = self.chem_batch_dict["nodes_features"]
        batch_chem.edge_index = self.chem_batch_dict["edges_index"]
        batch_chem.edge_attr = self.chem_batch_dict["edges_features"]
        batch_chem.batch = self.chem_batch_dict["graph_batch"]
        batch_prot.x = self.prot_batch_dict["nodes_features"]
        batch_prot.edge_index = self.prot_batch_dict["edges_index"]
        batch_prot.edge_attr = self.prot_batch_dict["edges_features"]
        batch_prot.batch = self.prot_batch_dict["graph_batch"]

        batch_chem.meta = torch.from_numpy(np.vstack(df_batch[chem_meta_key].map(self.chem_key_to_meta_dict))).type(
            torch.float32).to(self.device)
        batch_prot.meta = torch.from_numpy(np.vstack(df_batch[prot_meta_key].map(self.prot_key_to_meta_dict))).type(
            torch.float32).to(self.device)

        return self.y, batch_chem, batch_prot

    def reset(self):
        self.chem_batch_dict = {"node_features_len": self.chem_node_features_len,
                                "edge_features_len": self.chem_edge_features_len,
                                "n_nodes": 0,
                                "n_edges": 0,
                                "curr_node_index": 0,
                                "curr_edge_index": 0,
                                "edges_features": None}
        self.prot_batch_dict = {"node_features_len": self.prot_node_features_len,
                                "edge_features_len": self.prot_edge_features_len,
                                "n_nodes": 0,
                                "n_edges": 0,
                                "curr_node_index": 0,
                                "curr_edge_index": 0,
                                "edges_features": None}

        self.chem_gen_edge_features = False
        self.prot_gen_edge_features = False
        self.y = None

    def init_arrays(self, df_batch, chem_key, prot_key):
        for ind, row in df_batch.iterrows():
            self.get_n_elems(row[chem_key], self.chem_key_to_graph_dict, self.chem_batch_dict,
                             self.chem_gen_edge_features)
            self.get_n_elems(row[prot_key], self.prot_key_to_graph_dict, self.prot_batch_dict,
                             self.prot_gen_edge_features)

        self.get_zero_arrays(self.chem_batch_dict, self.chem_gen_edge_features)
        self.get_zero_arrays(self.prot_batch_dict, self.prot_gen_edge_features)

    @staticmethod
    def get_n_elems(key, graph_dict, batch_dict, gen_edge_features):
        features = graph_dict[key]

        assert features, f"None features for {key}"
        assert len(features[0]) > 1, f"N of nodes < 2 for {key}"
        assert len(features[1]) > 0, f"No edges for {key}"
        if gen_edge_features:
            assert len(features[1]) == len(features[2]), f"N of edges and edge features don't match for {key}"

        batch_dict["n_nodes"] += len(features[0])
        batch_dict["n_edges"] += len(features[1])

    @staticmethod
    def get_zero_arrays(batch_dict, gen_edge_features):
        batch_dict["nodes_features"] = np.zeros((batch_dict["n_nodes"], batch_dict["node_features_len"]),
                                                dtype=np.float32)
        batch_dict["edges_index"] = np.zeros((batch_dict["n_edges"], 2), dtype=np.int64)
        batch_dict["graph_batch"] = np.zeros(batch_dict["n_nodes"], dtype=np.int64)
        if gen_edge_features:
            batch_dict["edges_features"] = np.zeros((batch_dict["n_edges"], batch_dict["edge_features_len"]),
                                                    dtype=np.float32)

    @staticmethod
    def get_arrays(key, graph_dict, batch_dict, ind, gen_edge_features):
        features = graph_dict[key]

        cni = batch_dict["curr_node_index"]
        cei = batch_dict["curr_edge_index"]
        nni = len(features[0])
        nei = len(features[1])
        batch_dict["nodes_features"][cni: cni + nni] = np.array(features[0], dtype=np.float32)
        batch_dict["graph_batch"][cni: cni + nni] = np.full(nni, ind)
        batch_dict["edges_index"][cei: cei + nei] = np.array(features[1], dtype=np.int64) + cni
        if gen_edge_features:
            batch_dict["edges_features"][cei: cei + nei] = np.array(features[2], dtype=np.float32)
        batch_dict["curr_node_index"] += nni
        batch_dict["curr_edge_index"] += nei

    def conv_arrays_to_tensors(self, batch_dict, gen_edge_features):
        batch_dict["nodes_features"] = torch.from_numpy(batch_dict["nodes_features"]).type(torch.float32).to(
            self.device)
        batch_dict["edges_index"] = torch.from_numpy(batch_dict["edges_index"].T).type(torch.int64).to(self.device)
        batch_dict["graph_batch"] = torch.from_numpy(batch_dict["graph_batch"]).type(torch.int64).to(self.device)
        if gen_edge_features:
            batch_dict["edges_features"] = torch.from_numpy(batch_dict["edges_features"]).type(torch.float32).to(
                self.device)


def train(model, df_train, batch_generator, optimizer, criterion, epoch, key_kwargs, batch=32):
    y_true, y_pred, losses = [], [], []
    model.train()

    train_iters = math.ceil(df_train.shape[0] / batch)
    df_train_batches = np.array_split(df_train.sample(frac=1, random_state=epoch), train_iters)
    for df_train_batch in df_train_batches:
        y_train, chem_graph_train, prot_graph_train = batch_generator.get_batch(df_train_batch, **key_kwargs)

        optimizer.zero_grad()
        y_train_pred = model(chem_graph_train, prot_graph_train)

        train_loss = criterion(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        losses.append(train_loss.item())
        y_true.append(y_train)
        y_pred.append(y_train_pred)

    epoch_loss = sum(losses) / len(losses)
    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()

    return y_true, y_pred, epoch_loss


def val(model, df_val, batch_generator, criterion, key_kwargs, batch=32):
    y_true, y_pred, losses = [], [], []
    model.eval()

    val_iters = math.ceil(df_val.shape[0] / batch)
    df_val_batches = np.array_split(df_val, val_iters)
    for df_val_batch in df_val_batches:
        y_val, chem_graph_val, prot_graph_val = batch_generator.get_batch(df_val_batch, **key_kwargs)
        with torch.no_grad():
            y_val_pred = model(chem_graph_val, prot_graph_val)
            val_loss = criterion(y_val_pred, y_val)
        losses.append(val_loss.item())
        y_true.append(y_val)
        y_pred.append(y_val_pred)

    epoch_loss = sum(losses) / len(losses)
    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()
    return y_true, y_pred, epoch_loss


def get_metrics_reg(y_true, y_pred):
    metrics = dict()
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["medae"] = float(median_absolute_error(y_true, y_pred))
    metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    metrics["maxe"] = float(max_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["pearsonr"] = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    return metrics


def get_metrics_cls(y_true, y_pred, pred_transform=torch.sigmoid, threshold=0.5):
    if pred_transform is not None:
        y_pred = pred_transform(y_pred)
    y_pred_lbl = (y_pred >= threshold).type(torch.float32)

    metrics = dict()
    metrics["f1"] = float(f1_score(y_true, y_pred_lbl))
    metrics["precision"] = float(precision_score(y_true, y_pred_lbl, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred_lbl))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_lbl))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred_lbl))
    try:
        metrics["rocauc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        metrics["rocauc"] = np.nan
    try:
        precision_list, recall_list, thresholds = precision_recall_curve(y_true, y_pred)
        metrics["prauc"] = float(auc(recall_list, precision_list))
    except ValueError:
        metrics["prauc"] = np.nan

    return metrics


def train_val(model, optimizer, df_train, df_val, batch_generator, epochs, key_kwargs):
    criterion = torch.nn.MSELoss()
    epoch_to_metrics = dict()
    for epoch in tqdm(range(epochs)):
        epoch += 1

        y_true_train, y_pred_train, loss_train = train(model, df_train, batch_generator, optimizer, criterion, epoch,
                                                       key_kwargs)
        y_true_val, y_pred_val, loss_val = val(model, df_val, batch_generator, criterion, key_kwargs)

        if epoch % 10 == 0:
            metrics_train = get_metrics_reg(y_true_train, y_pred_train)
            metrics_val = get_metrics_reg(y_true_val, y_pred_val)
            metrics_val["ci"] = get_cindex(y_true_val.flatten().tolist(), y_pred_val.flatten().tolist())
            metrics_val["rm2"] = get_rm2(y_true_val.flatten().tolist(), y_pred_val.flatten().tolist())

            epoch_to_metrics[epoch] = dict()
            epoch_to_metrics[epoch]["metrics_train"] = metrics_train
            epoch_to_metrics[epoch]["metrics_val"] = metrics_val
            epoch_to_metrics[epoch]["loss_train"] = loss_train
            epoch_to_metrics[epoch]["loss_val"] = loss_val

            print("Epoch | ", epoch)
            print("Train | ", {k: round(v, 3) for k, v in epoch_to_metrics[epoch]["metrics_train"].items()},
                  "| loss: ", round(epoch_to_metrics[epoch]["loss_train"], 3))
            print("Test  | ", {k: round(v, 3) for k, v in epoch_to_metrics[epoch]["metrics_val"].items()},
                  "| loss: ", round(epoch_to_metrics[epoch]["loss_val"], 3))

    return epoch_to_metrics
