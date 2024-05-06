from transfer_to_continual.data import RandomMemorySetManager
from transfer_to_continual.managers import MnistManagerSplit, Cifar10ManagerSplit, Cifar100ManagerSplit
from transfer_to_continual.config import Config
from pathlib import Path
from itertools import zip_longest
import os
import csv

import wandb
import torch
import random
import numpy as np

import yaml
import argparse


def setup_wandb(config: Config):
    run_name = config.run_name
    experiment_tag = getattr(config, "experiment_tag", None)
    experiment_metadata_path = getattr(config, "experiment_metadata_path", None)
    tags = [experiment_tag] if experiment_tag is not None else []

    run = wandb.init(
        tags=tags,
        project=config.wandb_project_name,
        entity=config.wandb_profile,
        name=run_name,
        config=config.config_dict,
    )

    if experiment_metadata_path is not None:
        # Create csv storing run ida
        new_row = [run.path]
        file_exists = os.path.exists(experiment_metadata_path)
        # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
        with open(
            experiment_metadata_path, mode="a" if file_exists else "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(new_row)


def main(config: Config):
    if config.use_wandb:
        setup_wandb(config)

    random_seed = getattr(config, "random_seed", None)
    memory_set_manager = config.memory_set_manager(
        p=config.p, random_seed=random_seed
    )

    manager = config.learning_manager(
        memory_set_manager=memory_set_manager,
        use_wandb=config.use_wandb,
        transfer_metrics=config.transfer_metrics,
        model=config.model,
    )

    epochs = config.epochs
    num_tasks = manager.num_tasks

    # Train on first task
    final_accs = []
    final_task_spec_accs = []
    final_backward_transfers = []
    final_forward_transfers = []

    final_metrics = {"leep": [], "logme": [], "gbc": []}

    model_save_dir = getattr(config, "model_save_dir", None)
    model_load_dir = getattr(config, "model_load_dir", None)
    if model_load_dir is not None:
        print("Model load path given so loading model and not training")
        print("If this is unintended behaviour, remove model_load_dir from config")

    for i in range(num_tasks):
        metrics = dict()
        golden_model_accs = getattr(config, "golden_model_accs", None)
        if model_load_dir is not None:
            # Load model and run evaluation
            post_train_model_load_path = (
                model_load_dir / f"{config.model_name}_task_{i}.pt"
            )
            post_train_model = torch.load(post_train_model_load_path)
            if i > 0:
                # Can get pre training model and transfer metric value
                pre_train_model_load_path = (
                    model_load_dir / f"{config.model_name}_task_{i-1}.pt"
                )
                pre_train_model = torch.load(pre_train_model_load_path)
                metrics = manager.evaluate_transfer_metrics(model=pre_train_model)
            else:
                metrics = {
                    "leep": None,
                    "logme": None,
                    "gbc": None,
                }  # First task, no transfer metric values

            acc, task_spec_acc, backward_transfer, forward_transfer = manager.evaluate_task(
                model=post_train_model, golden_model_accs=golden_model_accs
            )
        else:
            golden_model: bool = getattr(config, "golden_model", None)
            if golden_model:
                print("Training golden model. Resetting manager model")
                # Reset the model architechture, and deactivate past tasks
                new_model = config.get_new_model()
                manager.set_model(new_model)
                manager.deactivate_past_tasks()
                # Check only 1 task is being trained
                assert(sum([task.active for task in manager.tasks]) == 1)

            # Train model from scratch
            if model_save_dir is not None:
                model_save_path = model_save_dir / f"{config.model_name}_task_{i}.pt"
            else:
                model_save_path = None

            print(f"Training on Task {i}")
            acc, task_spec_acc, backward_transfer, forward_transfer, metrics = manager.train(
                epochs=epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                use_memory_set=config.use_memory_set,
                model_save_path=model_save_path,
                golden_model_accs=golden_model_accs,
            )

        # Collect performance metrics
        final_accs.append(acc)
        final_task_spec_accs.append(task_spec_acc)
        final_backward_transfers.append(backward_transfer)
        final_forward_transfers.append(forward_transfer)
        for metric_name in metrics.keys():
            metric_val = metrics[metric_name]
            final_metrics[metric_name].append(metric_val)

        # Advance the task
        if i < num_tasks - 1:
            manager.next_task()

    # Log all final results
    tasks = list(range(num_tasks))
    data = [
        [task, final_acc, final_task_spec_acc, b_transfer, f_transfer, final_leep, final_logme, final_gbc]
        for task, final_acc, final_task_spec_acc, b_transfer, f_transfer, final_leep, final_logme, final_gbc in zip_longest(
            tasks,
            final_accs,
            final_task_spec_accs,
            final_backward_transfers,
            final_forward_transfers,
            final_metrics["leep"],
            final_metrics["logme"],
            final_metrics["gbc"],
        )
    ]
    table = wandb.Table(
        data=data, columns=["task_idx", "final_test_acc", "final_task_spec_acc", "final_test_backward_transfer", "final_test_forward_transfer", "leep", "logme", "gbc"]
    )  

    if config.use_wandb:
        wandb.log({"Metric Table": table})

        # Finish wandb run
        wandb.finish()

    # plot_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual training run")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file to run from.",
    )
    parser.add_argument(
        "--exp-path",
        type=str, 
        default="",
        help="If given then all for models with be prepended with exp-path"
    )
    args = parser.parse_args()

    with open(f"{args.config}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = Config(config_dict)
    config.set_experiment_path(args.exp_path)
    config.create_model_save_dirs()

    main(config)
