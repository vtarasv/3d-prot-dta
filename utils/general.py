import os
import pickle
import logging
import datetime

import numpy as np
import torch


log_path = os.path.join(os.path.dirname(__file__), "..", "log/")
os.makedirs(log_path, exist_ok=True)
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d_%H-%M-%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join(log_path, f"LOG_{now}.txt"))
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

c_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')
f_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class GraphBatchConstructorBase:
    def __init__(self, device):
        self.device = device

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


class GraphBatchConstructor(GraphBatchConstructorBase):
    def __init__(self, e1_key_to_graph_dict, device, e1_node_features_len=None, e1_edge_features_len=None):
        super(GraphBatchConstructor, self).__init__(device)
        self.e1_key_to_graph_dict = e1_key_to_graph_dict
        self.e1_node_features_len = e1_node_features_len
        self.e1_edge_features_len = e1_edge_features_len

        self.e1_gen_edge_features = False

        self.e1_batch_dict = {}

    def get_batch(self, df_batch, *, e1_key="", e1_gen_edge_features=True, **kwargs):
        self.reset()
        self.e1_gen_edge_features = e1_gen_edge_features

        df_batch.reset_index(drop=True, inplace=True)

        self.init_arrays(df_batch, e1_key)

        for ind, row in df_batch.iterrows():
            self.get_arrays(row[e1_key], self.e1_key_to_graph_dict, self.e1_batch_dict, ind,
                            self.e1_gen_edge_features)

        self.conv_arrays_to_tensors(self.e1_batch_dict, self.e1_gen_edge_features)

        class GraphBatch:
            pass
        batch_e1 = GraphBatch()
        batch_e1.x = self.e1_batch_dict["nodes_features"]
        batch_e1.edge_index = self.e1_batch_dict["edges_index"]
        batch_e1.edge_attr = self.e1_batch_dict["edges_features"]
        batch_e1.batch = self.e1_batch_dict["graph_batch"]

        return batch_e1

    def reset(self):
        self.e1_batch_dict = {"node_features_len": self.e1_node_features_len,
                              "edge_features_len": self.e1_edge_features_len,
                              "n_nodes": 0,
                              "n_edges": 0,
                              "curr_node_index": 0,
                              "curr_edge_index": 0,
                              "edges_features": None}

        self.e1_gen_edge_features = False

    def init_arrays(self, df_batch, e1_key):
        for ind, row in df_batch.iterrows():
            self.get_n_elems(row[e1_key], self.e1_key_to_graph_dict, self.e1_batch_dict,
                             self.e1_gen_edge_features)

        self.get_zero_arrays(self.e1_batch_dict, self.e1_gen_edge_features)


class GraphPairsBatchConstructor(GraphBatchConstructorBase):
    def __init__(self, e1_key_to_graph_dict, e2_key_to_graph_dict, device,
                 e1_node_features_len=None, e2_node_features_len=None,
                 e1_edge_features_len=None, e2_edge_features_len=None):
        super(GraphPairsBatchConstructor, self).__init__(device)
        self.e1_key_to_graph_dict = e1_key_to_graph_dict
        self.e2_key_to_graph_dict = e2_key_to_graph_dict
        self.e1_node_features_len = e1_node_features_len
        self.e2_node_features_len = e2_node_features_len
        self.e1_edge_features_len = e1_edge_features_len
        self.e2_edge_features_len = e2_edge_features_len

        self.e1_gen_edge_features = False
        self.e2_gen_edge_features = False

        self.e1_batch_dict = {}
        self.e2_batch_dict = {}

    def get_batch(self, df_batch, *, e1_key="", e2_key="", e1_gen_edge_features=True,
                  e2_gen_edge_features=True, **kwargs):
        self.reset()
        self.e1_gen_edge_features = e1_gen_edge_features
        self.e2_gen_edge_features = e2_gen_edge_features

        df_batch.reset_index(drop=True, inplace=True)

        self.init_arrays(df_batch, e1_key, e2_key)

        for ind, row in df_batch.iterrows():
            self.get_arrays(row[e1_key], self.e1_key_to_graph_dict, self.e1_batch_dict, ind,
                            self.e1_gen_edge_features)
            self.get_arrays(row[e2_key], self.e2_key_to_graph_dict, self.e2_batch_dict, ind,
                            self.e2_gen_edge_features)

        self.conv_arrays_to_tensors(self.e1_batch_dict, self.e1_gen_edge_features)
        self.conv_arrays_to_tensors(self.e2_batch_dict, self.e2_gen_edge_features)

        class GraphBatch:
            pass
        batch_e1, batch_e2 = GraphBatch(), GraphBatch()
        batch_e1.x = self.e1_batch_dict["nodes_features"]
        batch_e1.edge_index = self.e1_batch_dict["edges_index"]
        batch_e1.edge_attr = self.e1_batch_dict["edges_features"]
        batch_e1.batch = self.e1_batch_dict["graph_batch"]
        batch_e2.x = self.e2_batch_dict["nodes_features"]
        batch_e2.edge_index = self.e2_batch_dict["edges_index"]
        batch_e2.edge_attr = self.e2_batch_dict["edges_features"]
        batch_e2.batch = self.e2_batch_dict["graph_batch"]

        return batch_e1, batch_e2

    def reset(self):
        self.e1_batch_dict = {"node_features_len": self.e1_node_features_len,
                              "edge_features_len": self.e1_edge_features_len,
                              "n_nodes": 0,
                              "n_edges": 0,
                              "curr_node_index": 0,
                              "curr_edge_index": 0,
                              "edges_features": None}
        self.e2_batch_dict = {"node_features_len": self.e2_node_features_len,
                              "edge_features_len": self.e2_edge_features_len,
                              "n_nodes": 0,
                              "n_edges": 0,
                              "curr_node_index": 0,
                              "curr_edge_index": 0,
                              "edges_features": None}

        self.e1_gen_edge_features = False
        self.e2_gen_edge_features = False

    def init_arrays(self, df_batch, e1_key, e2_key):
        for ind, row in df_batch.iterrows():
            self.get_n_elems(row[e1_key], self.e1_key_to_graph_dict, self.e1_batch_dict,
                             self.e1_gen_edge_features)
            self.get_n_elems(row[e2_key], self.e2_key_to_graph_dict, self.e2_batch_dict,
                             self.e2_gen_edge_features)

        self.get_zero_arrays(self.e1_batch_dict, self.e1_gen_edge_features)
        self.get_zero_arrays(self.e2_batch_dict, self.e2_gen_edge_features)
