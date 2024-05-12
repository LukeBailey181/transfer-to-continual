from typing import Dict, Union
from .data import RandomMemorySetManager
from .managers import (
    MnistManagerSplit,
    Cifar10ManagerSplit,
    Cifar100ManagerSplit,
    Cifar100Manager20Split,
    Cifar10Full,
)
from pathlib import Path
from dataclasses import dataclass
from .models import MLP, CifarNet
import torch
import random
import numpy as np


@dataclass
class Config:
    def __init__(self, config_dict: Dict[str, Union[str, int, float]]):
        self.config_dict = config_dict

        # String run_name for wandb / logfiles
        self.run_name = (
            f"Manager.{config_dict['learning_manager']}_"
            f"MemorySetManager.{config_dict['memory_set_manager']}_p.{config_dict['p']}_"
            f"Model.{config_dict['model']['type']}"
        )

        if "random_seed" in config_dict:
            print("Seed given in config, setting deterministic run")
            random_seed = config_dict["random_seed"]

            # Set run to be deterministic
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        # Pass config into python
        for key, val in config_dict.items():
            if key == "memory_set_manager":
                if val == "random":
                    setattr(self, key, RandomMemorySetManager)
            elif key == "learning_manager":
                if val == "mnist_split":
                    setattr(self, key, MnistManagerSplit)
                elif val == "cifar10_split":
                    setattr(self, key, Cifar10ManagerSplit)
                elif val == "cifar100_5_split":
                    setattr(self, key, Cifar100ManagerSplit)
                elif val == "cifar100_20_split":
                    setattr(self, key, Cifar100Manager20Split)
                elif val == "cifar10_full":
                    setattr(self, key, Cifar10Full)
            elif key == "model":
                self.model_type = val["type"]
                self.model_params = val["params"]
                if self.model_type == "mlp":
                    setattr(self, key, MLP(**self.model_params))
                elif self.model_type == "cnn":
                    setattr(self, key, CifarNet(**self.model_params))
                    self.model.weight_init()
                else:
                    raise ValueError(
                        f"{self.model_type} model hasn't been implemented in this code"
                    )

                # Set model name 
                self.model_name = self.model_type
                for val in self.model_params.values():
                    self.model_name += f"_{val}"
            elif key == "model_load_dir":
                self.model_load_dir = Path(val)
            elif key == "model_save_dir":
                self.model_save_dir = Path(val)
            elif key == "experiment_metadata_path": 
                self.experiment_metadata_path = Path(val)
            elif key == "golden_model_accs":
                self.golden_model_accs = val

            else:
                setattr(self, key, val)

        if getattr(self, "model_load_dir", None) is not None:
            self.run_name = f"EVAL_" + self.run_name
        else:
            self.run_name = f"TRAIN_" + self.run_name

    def set_experiment_path(self, exp_path: str) -> None: 
        """Set the experiment path
        
        Args: 
            exp_path: Path that all model paths will be prepended with
        """
        self.exp_path = Path(exp_path)
        if getattr(self, "model_save_dir", None) is not None:
            self.model_save_dir = self.exp_path / self.model_save_dir
        if getattr(self, "model_load_dir", None) is not None:
            self.model_load_dir = self.exp_path / self.model_load_dir
        if getattr(self, "experiment_metadata_path", None) is not None:
            self.experiment_metadata_path = self.exp_path / self.experiment_metadata_path

    def create_model_save_dirs(self) -> None: 
        """Create paths create paths to save models"""
        # Create the directory and any missing parent directories

        if getattr(self, "model_save_dir", None) is not None: 
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "experiment_metadata_path", None) is not None:
            # Create any missing parent directories
            self.experiment_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            # Create the file if it doesn't exist
            self.experiment_metadata_path.touch(exist_ok=True)

    def get_new_model(self) -> None:
        """Generate a new model with the provided ARCH"""

        if self.model_type == "mlp":
            model = MLP(**self.model_params)
        elif self.model_type == "cnn":
            model = CifarNet(**self.model_params)
            self.model.weight_init()

        return model


def dict_to_config_string(config_dict: Dict[str, Union[str, int, float]]) -> str:
    # Convert config dict to string for wandb
    config_string = ""
    for key, val in config_dict.items():
        config_string += f"{key}.{val}_"

    if config_string[-1] == "_":
        config_string = config_string[:-1]
    return config_string
