import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GINConv, GATConv, GCNConv, NNConv, MFConv, GINEConv, GATv2Conv

from params import N_CHEM_NODE_FEAT, N_CHEM_EDGE_FEAT, N_PROT_NODE_FEAT, N_PROT_EDGE_FEAT, N_CHEM_ECFP


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
        activation = nn.ReLU()
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
                                          use_edges_features=False,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GATv2Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        super(GATv2Layers, self).__init__(trial, prefix=prefix + "_gatv2", gnn=GATv2Conv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=False,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GCNLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
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

            _nn = nn.Sequential(nn.Linear(_n_in, _n_out), nn.ReLU(), nn.Linear(_n_out, _n_out))
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
                 n_units_list=(64, 128, 256, 512, 1024,), **kwargs):
        super(GINLayers, self).__init__(trial, prefix=prefix + "_gin", gnn=GINConv,
                                        _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                        use_edges_features=False,
                                        layers_range=layers_range, n_units_list=n_units_list, **kwargs)


class GINELayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(64, 128, 256, 512, 1024,), **kwargs):
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
            nn_ = nn.Sequential(nn.Linear(self._edge_features_len, hidden_size), nn.ReLU(),
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
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
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


class Graph:
    pass


class GraphModel(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len):
        super(GraphModel, self).__init__()
        _n_out = None

        self._gnn_arch = trial.suggest_categorical(prefix + "_gnn_arch", ("single", "staked"))
        self._use_post_fc = trial.suggest_categorical(prefix + "_gnn_post_fc", (True,))
        prefix = prefix + "_" + self._gnn_arch

        if self._gnn_arch == "single":
            graph_models = ["GATv2Layers", "GCNLayers", "GINLayers", "GINELayers", "GMFLayers"]

            _graph_model_type = trial.suggest_categorical(prefix+"_graph_model_type", graph_models)
            self._graph_model = eval(_graph_model_type)(trial, prefix, _node_features_len=_node_features_len,
                                                        _edge_features_len=_edge_features_len)
            _n_out = self._graph_model.n_out
            self._pool = GraphPool(trial, prefix=prefix)
            _n_out = _n_out * self._pool.coef

        elif self._gnn_arch == "staked":
            _n_out = 0
            self.gat = GATv1Layers(trial, prefix, _node_features_len, _edge_features_len, heads_range=(1, 7),
                                   layers_range=(1, 3))
            self.gcn = GCNLayers(trial, prefix, self.gat.n_out)
            self.pool_gat_gcn = GraphPool(trial, prefix=prefix+"_gat_gcn")
            _n_out += self.gcn.n_out * self.pool_gat_gcn.coef

            self.use_gine = trial.suggest_categorical(prefix+"_use_gine", (True, False))
            if self.use_gine:
                self.gine = GINELayers(trial, prefix, _node_features_len, _edge_features_len)
                self.pool_gine = GraphPool(trial, prefix=prefix+"_gine")
                _n_out += self.gine.n_out * self.pool_gine.coef

            self.use_gqc = trial.suggest_categorical(prefix+"_use_gqc", (True, False))
            if self.use_gqc:
                self.gqc = GQCLayers(trial, prefix, _node_features_len, _edge_features_len)
                self.pool_gqc = GraphPool(trial, prefix=prefix+"_gqc")
                _n_out += self.gqc.n_out * self.pool_gqc.coef

            self.use_gmf = trial.suggest_categorical(prefix+"_use_gmf", (True, False))
            if self.use_gmf:
                self.gmf = GMFLayers(trial, prefix, _node_features_len)
                self.pool_gmf = GraphPool(trial, prefix=prefix+"_gmf")
                _n_out += self.gmf.n_out * self.pool_gmf.coef

        if self._use_post_fc:
            self._post_fc = FCLayers(trial, prefix+"_post", _n_out, layers_range=(1, 1),
                                     n_units_list=(256, 512, 1024, 2048))
            _n_out = self._post_fc.n_out

        self.n_out = _n_out

    def forward(self, graph):
        x = None

        if self._gnn_arch == "single":
            x = self._graph_model(graph)
            x = self._pool(x, graph.batch)

        elif self._gnn_arch == "staked":
            gat_out = self.gat(graph)
            graph_ = Graph()
            graph_.x = gat_out
            graph_.edge_index = graph.edge_index
            graph_.edge_attr = graph.edge_attr
            gat_gcn_out = self.gcn(graph_)
            x = self.pool_gat_gcn(gat_gcn_out, graph.batch)

            if self.use_gine:
                gine_out = self.gine(graph)
                gine_out = self.pool_gine(gine_out, graph.batch)
                x = torch.cat([x, gine_out], dim=1)
            if self.use_gqc:
                gqc_out = self.gqc(graph)
                gqc_out = self.pool_gqc(gqc_out, graph.batch)
                x = torch.cat([x, gqc_out], dim=1)
            if self.use_gmf:
                gmf_out = self.gmf(graph)
                gmf_out = self.pool_gmf(gmf_out, graph.batch)
                x = torch.cat([x, gmf_out], dim=1)

        if self._use_post_fc:
            x = self._post_fc(x)

        return x


class DTIProtGraphChemGraph(torch.nn.Module):
    def __init__(self, trial, chem_node_features_len=N_CHEM_NODE_FEAT, prot_node_features_len=N_PROT_NODE_FEAT,
                 chem_edge_features_len=N_CHEM_EDGE_FEAT, prot_edge_features_len=N_PROT_EDGE_FEAT):

        super(DTIProtGraphChemGraph, self).__init__()

        self.chem_graph_encoder = GraphModel(trial, "chem", chem_node_features_len, chem_edge_features_len)
        self.prot_graph_encoder = GraphModel(trial, "prot", prot_node_features_len, prot_edge_features_len)

        n_out = self.chem_graph_encoder.n_out + self.prot_graph_encoder.n_out

        self.fc = FCLayers(trial, "final", n_out, layers_range=(2, 3), n_units_list=(256, 512, 1024, 2048, 4096))
        self.out = torch.nn.Linear(self.fc.n_out, 1)

    def forward(self, data):
        chem_graph, prot_graph = data["e1_graph"], data["e2_graph"]

        chem_graph_out = self.chem_graph_encoder(chem_graph)
        prot_graph_out = self.prot_graph_encoder(prot_graph)
        x = torch.cat([chem_graph_out, prot_graph_out], dim=1)
        x = self.out(self.fc(x))

        return x


class DTIProtGraphChemECFP(torch.nn.Module):
    def __init__(self, trial, prot_node_features_len=N_PROT_NODE_FEAT, prot_edge_features_len=N_PROT_EDGE_FEAT,
                 chem_ecfp_len=N_CHEM_ECFP):

        super(DTIProtGraphChemECFP, self).__init__()

        self.use_chem_ecfp_post_fc = trial.suggest_categorical("chem_ecfp_post_fc", (True,))
        chem_ecfp_n_out = chem_ecfp_len
        if self.use_chem_ecfp_post_fc:
            self.chem_ecfp_post_fc = FCLayers(trial, "chem_ecfp_post", chem_ecfp_n_out, layers_range=(1, 1),
                                              n_units_list=(256, 512, 1024, 2048))
            chem_ecfp_n_out = self.chem_ecfp_post_fc.n_out

        self.prot_graph_encoder = GraphModel(trial, "prot", prot_node_features_len, prot_edge_features_len)

        n_out = chem_ecfp_n_out + self.prot_graph_encoder.n_out

        self.fc = FCLayers(trial, "final", n_out, layers_range=(2, 3), n_units_list=(256, 512, 1024, 2048, 4096))
        self.out = torch.nn.Linear(self.fc.n_out, 1)

    def forward(self, data):
        chem_ecfp, prot_graph = data["e1_fp"], data["e2_graph"]

        chem_ecfp_out = chem_ecfp
        if self.use_chem_ecfp_post_fc:
            chem_ecfp_out = self.chem_ecfp_post_fc(chem_ecfp_out)

        prot_graph_out = self.prot_graph_encoder(prot_graph)

        x = torch.cat([chem_ecfp_out, prot_graph_out], dim=1)
        x = self.out(self.fc(x))

        return x


class DTIProtGraphChemGraphECFP(torch.nn.Module):
    def __init__(self, trial, chem_node_features_len=N_CHEM_NODE_FEAT, prot_node_features_len=N_PROT_NODE_FEAT,
                 chem_edge_features_len=N_CHEM_EDGE_FEAT, prot_edge_features_len=N_PROT_EDGE_FEAT,
                 chem_ecfp_len=N_CHEM_ECFP):

        super(DTIProtGraphChemGraphECFP, self).__init__()

        self.use_chem_ecfp_post_fc = trial.suggest_categorical("chem_ecfp_post_fc", (True,))
        chem_ecfp_n_out = chem_ecfp_len
        if self.use_chem_ecfp_post_fc:
            self.chem_ecfp_post_fc = FCLayers(trial, "chem_ecfp_post", chem_ecfp_n_out, layers_range=(1, 1),
                                              n_units_list=(256, 512, 1024, 2048))
            chem_ecfp_n_out = self.chem_ecfp_post_fc.n_out

        self.chem_graph_encoder = GraphModel(trial, "chem", chem_node_features_len, chem_edge_features_len)
        self.prot_graph_encoder = GraphModel(trial, "prot", prot_node_features_len, prot_edge_features_len)

        n_out = chem_ecfp_n_out + self.chem_graph_encoder.n_out + self.prot_graph_encoder.n_out

        self.fc = FCLayers(trial, "final", n_out, layers_range=(2, 3), n_units_list=(256, 512, 1024, 2048, 4096))
        self.out = torch.nn.Linear(self.fc.n_out, 1)

    def forward(self, data):
        chem_ecfp, chem_graph, prot_graph = data["e1_fp"], data["e1_graph"], data["e2_graph"]

        chem_ecfp_out = chem_ecfp
        if self.use_chem_ecfp_post_fc:
            chem_ecfp_out = self.chem_ecfp_post_fc(chem_ecfp_out)

        chem_graph_out = self.chem_graph_encoder(chem_graph)
        prot_graph_out = self.prot_graph_encoder(prot_graph)

        x = torch.cat([chem_ecfp_out, chem_graph_out, prot_graph_out], dim=1)
        x = self.out(self.fc(x))

        return x
