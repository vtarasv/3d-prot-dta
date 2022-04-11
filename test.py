from utils import *
from models import *

import os
import pathlib
import torch
import json
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', '-b', type=str, required=True, choices=["davis", "kiba"],
                        help='name of the benchmark dataset')

    args = parser.parse_args()
    benchmark = args.benchmark

    data_path = pathlib.Path(__file__).parent.absolute().joinpath(f"data/{benchmark}/")
    res_path = pathlib.Path(__file__).parent.absolute().joinpath(f"results/")
    os.makedirs(res_path, exist_ok=True)

    prot_graph = load_pkl(data_path.joinpath("protid_to_graph.pkl"))
    lig_graph = load_pkl(data_path.joinpath("smiles_to_graph.pkl"))
    prot_meta = load_pkl(data_path.joinpath("protid_to_meta.pkl"))
    lig_meta = load_pkl(data_path.joinpath("smiles_iso_to_ecfp.pkl"))

    df = pd.read_csv(data_path.joinpath("full.csv"))

    test_fold = json.load(open(data_path.joinpath("folds/test_fold_setting1.txt")))
    train_folds = json.load(open(data_path.joinpath("folds/train_fold_setting1.txt")))

    df_train = df[~ df.index.isin(test_fold)]
    df_test = df[df.index.isin(test_fold)]

    batch_generator = BatchGenerator(lig_graph, prot_graph, lig_meta, prot_meta, DEVICE,
                                     chem_node_features_len=CHEM_NODE_SIZE, prot_node_features_len=PROT_NODE_SIZE,
                                     chem_edge_features_len=CHEM_EDGE_SIZE, prot_edge_features_len=PROT_EDGE_SIZE)

    torch.manual_seed(SEED)
    for i, fold_idxs in enumerate(train_folds):
        print(f"Fold {i} of {len(train_folds)}")
        model = DTI(trial=CustomTrial(hp=hp[benchmark]),
                    chem_meta_len=2048, prot_meta_len=meta_len[benchmark]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        df_train_ = df_train[~ df_train.index.isin(fold_idxs)]
        epoch_to_metrics = train_val(model, optimizer, df_train_, df_test, batch_generator,
                                     epochs=hp[benchmark]["epochs"],
                                     key_kwargs={"chem_graph_key": "smiles",
                                                 "prot_graph_key": "pid",
                                                 "chem_meta_key": "smiles_iso",
                                                 "prot_meta_key": "pid",
                                                 "label_key": "label",
                                                 "chem_gen_edge_features": True,
                                                 "prot_gen_edge_features": False})

        save_pkl(epoch_to_metrics, res_path.joinpath(f"{benchmark}_fold_{i}_results.pkl"))
