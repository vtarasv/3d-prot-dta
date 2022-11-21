import copy
import os

import torch
import numpy as np

from utils import logger, train_val, get_metrics_reg, save_pkl
from params import SEED, DEVICE, HP
from helpers import CustomTrial, CustomDataLoader, load_data
from models import DTIProtGraphChemGraphECFP

if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def run_dataset_test(dataset, model, folds):
    epochs = 700
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    for fold, idx_val in enumerate(val_folds):
        if fold not in folds:
            continue
        logger.info(f"Testing fold {fold} on {dataset} dataset")

        df_train = df_train_val[~ df_train_val.index.isin(idx_val)]

        test_dl = CustomDataLoader(df=df_test, batch_size=32, device=DEVICE,
                                   e1_key_to_graph=ligand_to_graph,
                                   e2_key_to_graph=protein_to_graph,
                                   e1_key_to_fp=ligand_to_ecfp,
                                   shuffle=False)
        train_dl = CustomDataLoader(df=df_train, batch_size=32, device=DEVICE,
                                    e1_key_to_graph=ligand_to_graph,
                                    e2_key_to_graph=protein_to_graph,
                                    e1_key_to_fp=ligand_to_ecfp,
                                    shuffle=True)

        model_copy = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        epoch_to_metrics = train_val(model=model_copy, optimizer=optimizer, criterion=criterion,
                                     train_dl=train_dl, val_dl=test_dl, epochs=epochs,
                                     score_fn=get_metrics_reg, fold=fold, verbose=True, with_rm2=True, with_ci=True)

        save_pkl(epoch_to_metrics, f"results/prot_graph-chem_graph_ecfp-{dataset}-fold_{fold}.pkl")


def main(folds, datasets):
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    os.makedirs("results/", exist_ok=True)
    if "davis" in datasets:
        run_dataset_test("davis", model, folds)
    if "kiba" in datasets:
        run_dataset_test("kiba", model, folds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, required=False, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--datasets', type=str, required=False, nargs='+',
                        default=["davis", "kiba"], choices=["davis", "kiba"])
    args = parser.parse_args()
    main(args.folds, args.datasets)
