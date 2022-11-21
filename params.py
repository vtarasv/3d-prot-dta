import torch

N_CHEM_NODE_FEAT = 23
N_PROT_NODE_FEAT = 41
N_CHEM_EDGE_FEAT = 6
N_PROT_EDGE_FEAT = 10
N_CHEM_ECFP = 2048

SEED = 47

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

HP = {
 'chem_ecfp_post_fc': True,
 'chem_ecfp_post_fc_dropout_0': 0.5,
 'chem_ecfp_post_fc_n_out_0': 256,
 'chem_ecfp_post_fc_use_bn': True,
 'chem_ecfp_post_n_fc_layers': 1,
 'chem_gnn_arch': 'single',
 'chem_gnn_post_fc': True,
 'chem_single_gatv2_activation': 'ReLU',
 'chem_single_gatv2_dropout_0': 0.0,
 'chem_single_gatv2_heads_0': 2,
 'chem_single_gatv2_n_layers': 1,
 'chem_single_gatv2_use_activation': True,
 'chem_single_gatv2_use_bn': False,
 'chem_single_graph_model_type': 'GATv2Layers',
 'chem_single_graph_pool_type': 'mean_add_max',
 'chem_single_post_fc_dropout_0': 0.3,
 'chem_single_post_fc_n_out_0': 1024,
 'chem_single_post_fc_use_bn': False,
 'chem_single_post_n_fc_layers': 1,
 'final_fc_dropout_0': 0.5,
 'final_fc_dropout_1': 0.5,
 'final_fc_dropout_2': 0.5,
 'final_fc_n_out_0': 2048,
 'final_fc_n_out_1': 1024,
 'final_fc_n_out_2': 512,
 'final_fc_use_bn': True,
 'final_n_fc_layers': 3,
 'prot_gnn_arch': 'single',
 'prot_gnn_post_fc': True,
 'prot_single_gin_activation': 'ReLU',
 'prot_single_gin_n_layers': 1,
 'prot_single_gin_n_out_0': 256,
 'prot_single_gin_use_activation': True,
 'prot_single_gin_use_bn': False,
 'prot_single_graph_model_type': 'GINLayers',
 'prot_single_graph_pool_type': 'mean_add_max',
 'prot_single_post_fc_dropout_0': 0.3,
 'prot_single_post_fc_n_out_0': 1024,
 'prot_single_post_fc_use_bn': True,
 'prot_single_post_n_fc_layers': 1
}
