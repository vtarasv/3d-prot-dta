import copy

import torch
import optuna
import numpy as np

from utils import logger, train_val, get_metrics_reg
from params import SEED, DEVICE
from helpers import CustomDataLoader, load_data
from models import DTIProtGraphChemGraphECFP

if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def run_dataset(trial, dataset, model):
    scores = []
    epochs = 150
    df_train_val, _, val_folds, _, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    for fold, idx_val in enumerate(val_folds):
        df_val = df_train_val[df_train_val.index.isin(idx_val)]
        df_train = df_train_val[~ df_train_val.index.isin(idx_val)]

        val_dl = CustomDataLoader(df=df_val, batch_size=32, device=DEVICE,
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
                                     train_dl=train_dl, val_dl=val_dl, epochs=epochs,
                                     score_fn=get_metrics_reg, fold=fold, val_nth_epoch=epochs)
        trial.set_user_attr(f"{dataset}_fold_{fold}_res_dict", epoch_to_metrics)
        score = epoch_to_metrics[epochs]["metrics_val"]["mse"]
        scores.append(score)
        if dataset == "davis" and score > 0.25:
            raise optuna.exceptions.TrialPruned()
        elif dataset == "kiba" and score > 0.18:
            raise optuna.exceptions.TrialPruned()

    return sum(scores) / len(scores)


def objective(trial):
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    try:
        # model = DTIProtGraphChemGraph(trial).to(DEVICE)
        # model = DTIProtGraphChemECFP(trial).to(DEVICE)
        model = DTIProtGraphChemGraphECFP(trial).to(DEVICE)
        score_davis = run_dataset(trial, "davis", model)
        score_kiba = run_dataset(trial, "kiba", model)
    except Exception as e:
        logger.debug("ERROR:", e, "| PARAMS:", trial.params)
        raise optuna.exceptions.TrialPruned()

    return (score_davis + score_kiba) / 2


if __name__ == "__main__":
    import argparse
    from optuna.samplers import RandomSampler, TPESampler

    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, required=True,
                        help='study name')
    parser.add_argument('--sampler', type=str, required=True, choices=["random", "tpe"],
                        help='optuna sampling algorithm')

    args = parser.parse_args()

    sampler = None
    if args.sampler == "random":
        sampler = RandomSampler()
    elif args.sampler == "tpe":
        sampler = TPESampler()

    optuna_study_path = "sqlite:///dta_tune.db"
    storage = optuna.storages.RDBStorage(optuna_study_path, heartbeat_interval=1)
    study = optuna.create_study(sampler=sampler, storage=storage, study_name=args.study,
                                direction="minimize", load_if_exists=True)
    logger.info(f"Tuning {args.study} | storage: {optuna_study_path} | sampler: {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=1000)
