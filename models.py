from params import *

import torch
import torch.nn as nn
from torch.nn import ReLU
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GINConv, GATConv, GCNConv, NNConv, MFConv, GINEConv, GATv2Conv


class CustomTrial:
    def __init__(self, hp):
        self.hp = hp
        self.suggest_int = self.get_from_dict
        self.suggest_float = self.get_from_dict
        self.suggest_categorical = self.get_from_dict
        self.suggest_discrete_uniform = self.get_from_dict
        self.suggest_loguniform = self.get_from_dict
        self.suggest_uniform = self.get_from_dict

    def get_from_dict(self, key, *args, **kwargs):
        try:
            param = self.hp[key]
        except KeyError:
            param = args[0]
            try:
                iter(param)
                param = param[0]
            except TypeError:
                pass
            print(f"No key {key} in hyper parameters, {param} will be used instead")

        return param


class GraphPool:
    def __init__(self, trial, prefix,
                 pool_types=("mean", "add", "max", "mean_add", "mean_max", "add_max", "mean_add_max")):
        self.coef_dict = {"mean": 1, "add": 1, "max": 1, "mean_add": 2, "mean_max": 2, "add_max": 2, "mean_add_max": 3}
        self.type_ = trial.suggest_categorical(prefix + "_graph_pool_type", pool_types)
        self.coef = self.coef_dict[self.type_]

    def __call__(self, _graph_out, _graph_batch):
        out = None
        if self.type_ == "mean":
            out = global_mean_pool(_graph_out, _graph_batch)
        elif self.type_ == "add":
            out = global_add_pool(_graph_out, _graph_batch)
        elif self.type_ == "max":
            out = global_max_pool(_graph_out, _graph_batch)
        elif self.type_ == "mean_add":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "add_max":
            out = torch.cat([global_add_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_add_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch),
                             global_max_pool(_graph_out, _graph_batch)], dim=1)
        return out


class FCLayers(torch.nn.Module):
    def __init__(self, trial, prefix, in_features, layers_range=(2, 3), n_units_list=(128, 256, 512, 1024, 2048, 4096),
                 dropout_range=(0.1, 0.7), **kwargs):
        super(FCLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self.in_features = in_features
        self.layers = None
        self.n_out = None

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list, dropout_range=dropout_range)

    def get_layers(self, layers_range, n_units_list, dropout_range):

        in_features = self.in_features
        fc_layers = []
        n_fc_layers = self.trial.suggest_int(self.prefix + "_n_fc_layers", layers_range[0], layers_range[1])
        activation = ReLU()
        use_batch_norm = self.trial.suggest_categorical(self.prefix + "_fc_use_bn", (True, False))
        out_features = None
        for i in range(n_fc_layers):
            out_features = self.trial.suggest_categorical(self.prefix + f"_fc_n_out_{i}", n_units_list)
            dropout = self.trial.suggest_float(self.prefix + f"_fc_dropout_{i}", dropout_range[0], dropout_range[1])

            fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(dropout))

            in_features = out_features

        self.layers = nn.Sequential(*fc_layers)
        self.n_out = out_features

    def forward(self, x):
        return self.layers(x)


class GNNLayers(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len=None, _edge_features_len=None, use_edges_features=False):
        super(GNNLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self._node_features_len = _node_features_len
        self._edge_features_len = _edge_features_len
        self.use_edges_features = use_edges_features

        self.activation = None
        self.layers_list = None
        self.bn_list = None
        self.n_out = None

    def forward(self, data, **kwargs):
        _graph_out = data.x
        _edges_index = data.edge_index
        _edges_features = data.edge_attr if self.use_edges_features else None

        for _nn, _bn in zip(self.layers_list, self.bn_list):
            _graph_out = _nn(_graph_out, _edges_index, edge_attr=_edges_features) if self.use_edges_features else _nn(
                _graph_out, _edges_index)
            if _bn is not None:
                _graph_out = _bn(_graph_out)
            if self.activation is not None:
                _graph_out = self.activation(_graph_out)

        return _graph_out


class GATLayers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 heads_range, dropout_range, **kwargs):
        super(GATLayers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                        _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn

        self.get_layers(layers_range=layers_range, heads_range=heads_range, dropout_range=dropout_range)

    def get_layers(self, layers_range, heads_range, dropout_range):
        _node_features_len = self._node_features_len
        _edge_features_len = self._edge_features_len

        self.use_edges_features = self.trial.suggest_categorical(self.prefix + "_use_edges_features",
                                                                 (True, False)) if self.use_edges_features else False
        if self.use_edges_features:
            _edge_features_fill = self.trial.suggest_categorical(self.prefix + "_edge_features_fill",
                                                                 ("mean", "add", "max", "mul", "min"))
        else:
            _edge_features_len = None
            _edge_features_fill = "mean"

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []
        _all_heads = 1

        for i in range(_n_layers):
            _heads = self.trial.suggest_int(self.prefix + f"_heads_{i}", heads_range[0], heads_range[1])
            _dropout = self.trial.suggest_float(self.prefix + f"_dropout_{i}", dropout_range[0], dropout_range[1])

            if self.use_edges_features:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout, edge_dim=_edge_features_len, fill_value=_edge_features_fill)
            else:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout)
            _layers_list.append(_gnn)

            _all_heads = _all_heads * _heads

            if use_bn:
                bn = nn.BatchNorm1d(_node_features_len * _all_heads)
                _bn_layers_list.append(bn)

        self.n_out = _node_features_len * _all_heads
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GATv1Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        super(GATv1Layers, self).__init__(trial, prefix=prefix + "_gatv1", gnn=GATConv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=True,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GATv2Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        # TODO: use_edges_features=True for GATv2Conv when edge_attrs are applicable for that gnn
        super(GATv2Layers, self).__init__(trial, prefix=prefix + "_gatv2", gnn=GATv2Conv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=False,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GCNLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(128, 256, 512, 1024,),
                 **kwargs):
        super(GCNLayers, self).__init__(trial, prefix + "_gcn", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = GCNConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GIN_Layers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 n_units_list, **kwargs):
        super(GIN_Layers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                         _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _nn = nn.Sequential(nn.Linear(_n_in, _n_out), ReLU(), nn.Linear(_n_out, _n_out))
            _gnn = self.gnn(_nn, edge_dim=self._edge_features_len) if self.use_edges_features else self.gnn(_nn)
            _layers_list.append(_gnn)
            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GINLayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(128, 256, 512, 1024,), **kwargs):
        super(GINLayers, self).__init__(trial, prefix=prefix + "_gin", gnn=GINConv,
                                        _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                        use_edges_features=False,
                                        layers_range=layers_range, n_units_list=n_units_list, **kwargs)


class GINELayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(128, 256, 512, 1024,), **kwargs):
        super(GINELayers, self).__init__(trial, prefix=prefix + "_gine", gnn=GINEConv,
                                         _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                         use_edges_features=True,
                                         layers_range=layers_range, n_units_list=n_units_list, **kwargs)


class GQCLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3),
                 n_units_list=(16, 32, 64), **kwargs):
        super(GQCLayers, self).__init__(trial, prefix + "_gqc", _node_features_len=_node_features_len,
                                        _edge_features_len=_edge_features_len, use_edges_features=True)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)
            hidden_size = round((_n_in * _n_out) / 2)
            nn_ = nn.Sequential(nn.Linear(self._edge_features_len, hidden_size), ReLU(),
                                nn.Linear(hidden_size, _n_in * _n_out))
            _gnn = NNConv(_n_in, _n_out, nn_)
            _layers_list.append(_gnn)

            if use_bn:
                bn = torch.nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GMFLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(128, 256, 512, 1024,),
                 **kwargs):
        super(GMFLayers, self).__init__(trial, prefix + "_gmf", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = MFConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


def get_sequential_graph_layers(trial, prefix, graph_models, kwargs):
    n_graph_models = trial.suggest_int(f"{prefix}_n_graph_models", 1, 2)
    graph_model_1_type = trial.suggest_categorical(f"{prefix}_graph_model_1_type", graph_models)
    graph_models_left = graph_models.copy()
    graph_models_left.remove(graph_model_1_type)
    if graph_model_1_type != "GQCLayers":
        graph_models_left.remove("GQCLayers")
    graph_model_2_type = trial.suggest_categorical(
        f"{prefix}_graph_model_2_type_from_" + "_".join([i[:-6] for i in graph_models_left]),
        graph_models_left) if n_graph_models > 1 else ""

    kwargs["heads_range"] = (1, 3)
    graph_model_1 = eval(graph_model_1_type)(trial, prefix, **kwargs)
    kwargs["_node_features_len"] = graph_model_1.n_out
    graph_model_2 = eval(graph_model_2_type)(trial, prefix, **kwargs) if graph_model_2_type else None

    return graph_model_1, graph_model_2


class DTI(torch.nn.Module):
    def __init__(self, trial, chem_node_features_len=CHEM_NODE_SIZE, prot_node_features_len=PROT_NODE_SIZE,
                 chem_edge_features_len=CHEM_EDGE_SIZE, prot_edge_features_len=PROT_EDGE_SIZE,
                 chem_meta_len=None, prot_meta_len=None):

        super(DTI, self).__init__()

        graph_models = ["GATv1Layers", "GATv2Layers", "GCNLayers", "GINLayers", "GINELayers", "GQCLayers", "GMFLayers"]

        self.use_chem_post_fc = trial.suggest_categorical("chem_post_fc", (True, False))
        self.use_prot_post_fc = trial.suggest_categorical("prot_post_fc", (True, False))
        self.use_chem_meta = trial.suggest_categorical("chem_use_meta", (True, False))

        chem_kwargs = {"_node_features_len": chem_node_features_len,
                       "_edge_features_len": chem_edge_features_len}
        self.chem_graph_model_1, self.chem_graph_model_2 = \
            get_sequential_graph_layers(trial, "chem", graph_models, chem_kwargs)
        chem_n_out = self.chem_graph_model_2.n_out if self.chem_graph_model_2 else self.chem_graph_model_1.n_out
        self.chem_pool = GraphPool(trial, prefix="chem", pool_types=("mean_add", "mean_max", "add_max", "mean_add_max"))
        chem_n_out = chem_n_out * self.chem_pool.coef

        if self.use_chem_meta:
            chem_n_out = chem_n_out + chem_meta_len
        if self.use_chem_post_fc:
            self.chem_post_fc = FCLayers(trial, "chem_post", chem_n_out, layers_range=(1, 2),
                                         n_units_list=(128, 256, 512, 1024, 2048, 4096))
            chem_n_out = self.chem_post_fc.n_out

        prot_kwargs = {"_node_features_len": prot_node_features_len,
                       "_edge_features_len": prot_edge_features_len}
        self.prot_graph_model_1, self.prot_graph_model_2 = \
            get_sequential_graph_layers(trial, "prot", graph_models, prot_kwargs)
        prot_n_out = self.prot_graph_model_2.n_out if self.prot_graph_model_2 else self.prot_graph_model_1.n_out
        self.prot_pool = GraphPool(trial, prefix="prot", pool_types=("mean_add", "mean_max", "add_max", "mean_add_max"))
        prot_n_out = prot_n_out * self.prot_pool.coef + prot_meta_len

        if self.use_prot_post_fc:
            self.prot_post_fc = FCLayers(trial, "prot_post", prot_n_out, layers_range=(1, 2),
                                         n_units_list=(128, 256, 512, 1024, 2048, 4096))
            prot_n_out = self.prot_post_fc.n_out

        n_out = chem_n_out + prot_n_out

        self.fc = FCLayers(trial, "final", n_out, layers_range=(2, 3),
                           n_units_list=(128, 256, 512, 1024, 2048, 4096, 8192))
        self.out = torch.nn.Linear(self.fc.n_out, 1)

    def forward(self, chem_graph, prot_graph):

        chem_graph_out = self.chem_graph_model_1(chem_graph)
        if self.chem_graph_model_2:
            chem_graph.x = chem_graph_out
            chem_graph_out = self.chem_graph_model_2(chem_graph)
        chem_graph_out = self.chem_pool(chem_graph_out, chem_graph.batch)
        if self.use_chem_meta:
            chem_graph_out = torch.cat([chem_graph_out, chem_graph.meta], dim=1)
        if self.use_chem_post_fc:
            chem_graph_out = self.chem_post_fc(chem_graph_out)

        prot_graph_out = self.prot_graph_model_1(prot_graph)
        if self.prot_graph_model_2:
            prot_graph.x = prot_graph_out
            prot_graph_out = self.prot_graph_model_2(prot_graph)
        prot_graph_out = self.prot_pool(prot_graph_out, prot_graph.batch)
        prot_graph_out = torch.cat([prot_graph_out, prot_graph.meta], dim=1)
        if self.use_prot_post_fc:
            prot_graph_out = self.prot_post_fc(prot_graph_out)

        x = torch.cat([chem_graph_out, prot_graph_out], dim=1)
        x = self.out(self.fc(x))

        return x
