import torch

CHEM_NODE_SIZE = 14
CHEM_EDGE_SIZE = 6

PROT_NODE_SIZE = 18
PROT_EDGE_SIZE = 9

SEED = 47

if torch.cuda.is_available():
    print(f"GPU will be used for training ({torch.cuda.get_device_name()})")
    DEVICE = torch.device("cuda")
else:
    print("CPU will be used for training")
    DEVICE = torch.device("cpu")

meta_len = {"davis": 12, "kiba": 108}

hp = dict()

hp["davis"] = {'chem_gcn_activation': 'ReLU',
               'chem_gcn_n_layers': 2,
               'chem_gcn_n_out_0': 128,
               'chem_gcn_n_out_1': 256,
               'chem_gcn_use_activation': True,
               'chem_gcn_use_bn': False,
               'chem_gine_n_layers': 2,
               'chem_gine_n_out_0': 128,
               'chem_gine_n_out_1': 128,
               'chem_gine_use_activation': False,
               'chem_gine_use_bn': False,
               'chem_graph_model_1_type': 'GCNLayers',
               'chem_graph_model_2_type_from_GATv1_GATv2_GIN_GINE_GMF': 'GINELayers',
               'chem_graph_pool_type': 'mean_add_max',
               'chem_n_graph_models': 2,
               'chem_post_fc': True,
               'chem_post_fc_dropout_0': 0.45,
               'chem_post_fc_n_out_0': 512,
               'chem_post_fc_use_bn': False,
               'chem_post_n_fc_layers': 1,
               'chem_use_meta': True,
               'final_fc_dropout_0': 0.16,
               'final_fc_dropout_1': 0.23,
               'final_fc_n_out_0': 2048,
               'final_fc_n_out_1': 1024,
               'final_fc_use_bn': True,
               'final_n_fc_layers': 2,
               'prot_gcn_activation': 'ReLU',
               'prot_gcn_n_layers': 3,
               'prot_gcn_n_out_0': 1024,
               'prot_gcn_n_out_1': 256,
               'prot_gcn_n_out_2': 512,
               'prot_gcn_use_activation': True,
               'prot_gcn_use_bn': False,
               'prot_graph_model_1_type': 'GCNLayers',
               'prot_graph_pool_type': 'mean_add',
               'prot_n_graph_models': 1,
               'prot_post_fc': False,
               'epochs': 400}

hp["kiba"] = {'chem_gatv2_activation': 'ReLU',
              'chem_gatv2_dropout_0': 0.2,
              'chem_gatv2_dropout_1': 0.1,
              'chem_gatv2_heads_0': 3,
              'chem_gatv2_heads_1': 1,
              'chem_gatv2_n_layers': 2,
              'chem_gatv2_use_activation': True,
              'chem_gatv2_use_bn': False,
              'chem_graph_model_1_type': 'GATv2Layers',
              'chem_graph_pool_type': 'mean_add',
              'chem_n_graph_models': 1,
              'chem_post_fc': True,
              'chem_post_fc_dropout_0': 0.2,
              'chem_post_fc_n_out_0': 512,
              'chem_post_fc_use_bn': False,
              'chem_post_n_fc_layers': 1,
              'chem_use_meta': True,
              'final_fc_dropout_0': 0.3,
              'final_fc_dropout_1': 0.6,
              'final_fc_dropout_2': 0.5,
              'final_fc_n_out_0': 1024,
              'final_fc_n_out_1': 512,
              'final_fc_n_out_2': 256,
              'final_fc_use_bn': False,
              'final_n_fc_layers': 3,
              'prot_gin_activation': 'LeakyReLU',
              'prot_gin_n_layers': 3,
              'prot_gin_n_out_0': 512,
              'prot_gin_n_out_1': 256,
              'prot_gin_n_out_2': 256,
              'prot_gin_use_activation': True,
              'prot_gin_use_bn': False,
              'prot_graph_model_1_type': 'GINLayers',
              'prot_graph_pool_type': 'mean_add',
              'prot_n_graph_models': 1,
              'prot_post_fc': False,
              'epochs': 300}
