from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set, Dict
from pathlib import Path

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import MemorySetManager
from .models import MLP, MNLIST_MLP_ARCH, CifarNet, CIFAR10_ARCH, CIFAR100_ARCH
from .training_utils import (
    MNIST_FEATURE_SIZE,
    convert_torch_dataset_to_tensor,
    plot_cifar_image,
)
from .tasks import Task
from .transfer_metrics.LEEP import LEEP
from .transfer_metrics.LogME import LogME
from .transfer_metrics.GBC import GBC
from .transfer_metrics.GBC_all import GBC_all

DEBUG = False

# Check for M1 Mac MPS (Apple Silicon GPU) support
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Using M1 Mac")
    DEVICE = torch.device("mps")
# Check for CUDA support (NVIDIA GPU)
elif torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
# Default to CPU if neither is available
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")

class ContinualLearningManager(ABC):
    """Class that manages continual learning training.

    For each different set of tasks, a different manager should be made.
    For example, one manager for MnistSplit, and one for CifarSplit.
    As much shared functionality as possibly should be abstracted into this
    base class.
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        """
        Args:
            memory_set_manager: The memory set manager to use to optionally create memory set.
            model: Model to be trained
            dataset_path: Path to the directory where the dataset is stored. TODO change this
            use_wandb: Whether to use wandb to log training.
        """
        self.use_wandb = use_wandb

        self.model = model
        self.transfer_metrics = transfer_metrics
        self.transfer_metric_dict = {"leep": LEEP, "logme": LogME(), "gbc": GBC, "gbc_all": GBC_all}

        train_x, train_y, test_x, test_y = self._load_dataset(dataset_path=dataset_path)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.memory_set_manager = memory_set_manager

        self.tasks = self._init_tasks()  # List of all tasks
        self.label_to_task_idx = dict()

        # Update label_to_task_idx
        for i, task in enumerate(self.tasks):
            for label in task.task_labels:
                assert label not in self.label_to_task_idx
                self.label_to_task_idx[label] = i

        self.num_tasks = len(self.tasks)
        self.task_index = (
            0  # Index of the current task, all tasks <= task_index are active
        )

        # Performance metrics
        self.R_full = torch.ones(self.num_tasks, self.num_tasks) * -1   # Cutting logits :self.task_index+1
        self.R_splice = torch.ones(self.num_tasks, self.num_tasks) * -1
        

    @abstractmethod
    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them"""
        pass

    @abstractmethod
    def _load_dataset(
        self,
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n 1"],
        Float[Tensor, "m f"],
        Float[Tensor, "m 1"],
    ]:
        """Load full dataset for all tasks"""
        pass

    @torch.no_grad()
    def evaluate_transfer_metrics(
        self,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Evaluate models for transfer to current task as per self.task_index

        Args:
            model: Model to evaluate. If none then 
                use self.model
        """

        if model is None:
            model = self.model

        metric_vals = dict.fromkeys(self.transfer_metrics)
        current_labels: List[int] = list(self._get_current_labels())

        print("Evaluating metric values")
        model.eval()
        for metric in self.transfer_metrics:
            if metric == "leep":
                [x, y] = (
                    self.tasks[self.task_index].train_x,
                    self.tasks[self.task_index].train_y,
                )
                x = x.to(DEVICE)
                pseudo_source_label, target_label = (
                    model(x).detach().to("cpu"),
                    y.numpy(),
                )

                pseudo_source_label = torch.softmax(
                    pseudo_source_label, dim=-1
                ).numpy()

                leep = self.transfer_metric_dict[metric]
                metric_vals[metric] = leep(pseudo_source_label, target_label).item()

                # leep_val = metric_vals[metric].item()
                # if self.use_wandb and leep_val:
                #     wandb.log({f"leep_task_idx_{self.task_index}": leep_val})

            if metric == "logme":
                [x, y] = (
                    self.tasks[self.task_index].train_x,
                    self.tasks[self.task_index].train_y,
                )
                x = x.to(DEVICE)
                f, y = model(x).detach().to("cpu").numpy(), y.numpy()

                print(type(f), type(y))
                print(f.shape, y.shape)

                logme = self.transfer_metric_dict[metric]
                metric_vals[metric] = logme.fit(f, y).item()

            if metric == "gbc":
                [x, y] = (
                    self.tasks[self.task_index].train_x,
                    self.tasks[self.task_index].train_y,
                )

                f_s, y = (
                    model.forward(x, return_preactivations=True)[1],
                    y.numpy(),
                )

                scaling=StandardScaler()
                
                # Use fit and transform method 
                scaling.fit(f_s)
                Scaled_data=scaling.transform(f_s)

                # print(Scaled_data)
                
                principal=PCA(n_components=64)
                principal.fit(Scaled_data)

                f_spca=principal.transform(Scaled_data)

                f_s = torch.DoubleTensor(f_spca)

                gbc = self.transfer_metric_dict[metric]
                metric_vals[metric] = gbc(f_s, y, current_labels).item()

            if metric == "gbc_all":
                [x_old, y_old] = (torch.tensor([]), torch.tensor([]))
                [x, y] = (
                    torch.cat((x_old, self.tasks[self.task_index].train_x), dim=-1),
                    torch.cat((y_old, self.tasks[self.task_index].train_y), dim=-1),
                )

                f_s, y = (
                    model.forward(x, return_preactivations=True)[1],
                    y.numpy(),
                )

                print(f_s, y)
                scaling=StandardScaler()
                
                # Use fit and transform method 
                scaling.fit(f_s)
                Scaled_data=scaling.transform(f_s)

                # print(Scaled_data)
                
                principal=PCA(n_components=64)
                principal.fit(Scaled_data)

                f_spca=principal.transform(Scaled_data)

                f_s = torch.DoubleTensor(f_spca)

                print(f_s)

                gbc_all = self.transfer_metric_dict[metric]
                metric_vals[metric] = gbc_all(f_s, y, current_labels)

                x_old, y_old = x, y

            val = metric_vals[metric]
            if self.use_wandb and val is not None:
                wandb.log({f"{metric}_task_idx_{self.task_index}": val})

        # Return model to training mode
        model.train()
        
        return metric_vals

    @torch.no_grad()
    def evaluate_task(
        self,
        test_dataloader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None,
        golden_model_accs : Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, float]:
        """Evaluate models on current task.
        
        Args:
            test_dataloader: Dataloader containing task data. If None 
                then test_dataloader up to and including current task 
                is used through self._get_task_dataloaders.
            model: Model to evaluate. If None then use self.model.
            golden_model_vals: Accuracy of model trained only on 
                data and classification task for individual tasks.
                Used for calculating forward transfer.
        """

        if model is None:
            model = self.model
        if test_dataloader is None:
            _, test_dataloader = self._get_task_dataloaders(
                use_memory_set=False, batch_size=64
            )

        current_labels: List[int] = list(self._get_current_labels())
        model.eval()

        # Record RTj values accuracy of the model on task j after training on task T
        # Want to get RTi and loop over i values from 1 to T
        total_correct = 0
        total_correct_splice = 0
        total_examples = 0
        num_tasks = self._get_num_activate_tasks()

        task_wise_correct = [0] * num_tasks
        task_wise_correct_splice = [0] * num_tasks
        task_wise_examples = [0] * num_tasks 

        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            # -----GET VALUES WITH FULL SPLICE----- # 
            outputs_splice = outputs[:, current_labels]

            # Note these task indexes are not the true task indexes, 
            # and instead INDEXES into self._get_current_labels()
            task_idxs = torch.tensor(
                [self.label_to_task_idx[y.item()] for y in batch_y]
            )
            correct = torch.argmax(outputs_splice, dim=1) == batch_y

            for i in range(num_tasks): 
                task_wise_correct[i] += torch.sum(correct[task_idxs == i]).item()
                task_wise_examples[i] += torch.sum(task_idxs == i).item()
                # Do the task splicing here, just for that task

            total_correct += (
                (torch.argmax(outputs_splice, dim=1) == batch_y).sum().item()
            )
            total_examples += batch_x.shape[0]

            # -----GET VALUES WITH TASK SPECIFIC SPLICE----- # 
            for label, pred in zip(batch_y, outputs):
                task_idx = self.label_to_task_idx[label.item()]
                label_splice = torch.tensor(sorted(list(self.tasks[task_idx].task_labels)))
                outputs_splice = pred[label_splice]
                argmax_val = torch.argmax(outputs_splice)
                pred = label_splice[argmax_val].item()

                correct_splice = pred == label
                task_wise_correct_splice[task_idx] += correct_splice.item()
                total_correct_splice += correct_splice.item()
            

        task_accs = [cor/total for cor, total in zip(task_wise_correct, task_wise_examples)]
        task_accs_splice = [cor/total for cor, total in zip(task_wise_correct_splice, task_wise_examples)]


        #R_ji means we are on task j and evaluating on task i
        # Let T be the current task
        # R_Tj = task_accs[j]
        if num_tasks == self.task_index + 1:

            # Backward and forward transfer
            T = self.task_index
            backward_transfer = 0
            forward_transfer = 0
            for i in range(T+1):
                # Can use R_full later, for now use splice value
                self.R_full[T, i] = task_accs[i]
                self.R_splice[T, i] = task_accs_splice[i]

                R_Ti = self.R_splice[T, i].item()
                R_ii = self.R_splice[i, i].item()

                assert(R_Ti != -1 and R_ii != -1)
                
                # Accumulate as long as i not final task
                if i != T:
                    backward_transfer += R_Ti - R_ii
                if i != 0 and golden_model_accs is not None:
                    if i not in golden_model_accs:
                        raise ValueError(f"Golden model accuracy not found for task {i}")
                    R_sep = golden_model_accs[i]
                    forward_transfer += R_ii - R_sep

            # Can only do backward transfer for tasks after first
            if T == 0:
                backward_transfer = -1
                forward_transfer = -1
            else:
                backward_transfer /= T
                forward_transfer /= T
        else:
            # Not training on previous tasks, so backwards and forwards
            # transfer cannot be calculated
            backward_transfer = -1
            forward_transfer = -1
        if golden_model_accs is None:
            # Forward transfer was not computed
            forward_transfer = -1

        test_acc = total_correct / total_examples
        test_task_spec_acc = total_correct_splice / total_examples
        if self.use_wandb:
            wandb.log(
                {
                    f"test_acc_task_idx_{self.task_index}": test_acc,
                    f"test_task_spec_acc_task_idx_{self.task_index}": test_task_spec_acc,
                    f"backward_transfer_task_idx_{self.task_index}": backward_transfer,
                    f"forward_transfer_task_idx_{self.task_index}": forward_transfer,
                }
            )

        model.train()

        return test_acc, test_task_spec_acc, backward_transfer, forward_transfer

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.01,
        use_memory_set: bool = False,
        model_save_path : Optional[Path] = None,
        golden_model_accs : Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Train on all tasks with index <= self.task_index

        Args:
            epochs: Number of epochs to train for.
            batch_size: Batch size to use for training.
            lr: Learning rate to use for training.
            use_memory_set: True then tasks with index < task_index use memory set,
                otherwise they use the full training set.
            save_model_path: If not None, then save the model to this path.

        Returns:
            Final test accuracy.
        """

        def get_next_batch(iterators, iterator_idx, dataloader):
            iterator = iterators[iterator_idx]
            try:
                # Try to fetch the next item
                batch = next(iterator)
            except StopIteration:
                # If StopIteration is raised, start the DataLoader from the beginning
                new_iterator = iter(dataloader)
                batch = next(new_iterator)
                # Replace iterator with new one
                iterators[iterator_idx] = new_iterator
            return batch

        self.model.train()
        self.model.to(DEVICE)

        train_dataloaders, test_dataloader = self._get_task_dataloaders(
            use_memory_set, batch_size
        )

        current_labels: List[int] = list(self._get_current_labels())

        # Train on batches
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification tasks
        optimizer = Adam(self.model.parameters(), lr=lr)

        # Calculate metric value 
        metric_vals: Dict[str, float] = self.evaluate_transfer_metrics()

        self.model.train()

        terminal_dataloader: DataLoader = train_dataloaders[-1]

        memory_dataloaders: List[DataLoader] = train_dataloaders[:-1]
        memory_iterators = [iter(dataloader) for dataloader in memory_dataloaders]

        for _ in tqdm(range(epochs)):
            for terminal_batch_x, terminal_batch_y in terminal_dataloader:
                
                # Get the data from memory dataloaders
                memory_batches = [
                    get_next_batch(memory_iterators, iterator_idx, dataloader) for iterator_idx, dataloader in enumerate(memory_dataloaders)
                ]

                # Concatenate with memory batches if there are any
                if memory_batches:
                    memory_batch_x = torch.cat([batch[0] for batch in memory_batches])
                    memory_batch_y = torch.cat([batch[1] for batch in memory_batches])
                    batch_x = torch.cat([terminal_batch_x, memory_batch_x])
                    batch_y = torch.cat([terminal_batch_y, memory_batch_y])
                else:
                    batch_x = terminal_batch_x
                    batch_y = terminal_batch_y

                if DEBUG and self.task_index == 1:
                    # TODO make this generic to MNIST or CIFAR
                    # TODO make degbugging config flag
                    print(batch_y)
                    plot_cifar_image(batch_x)

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(batch_x)

                outputs = outputs[
                    :, current_labels
                ]  # Only select outputs for current labels

                # Cutting outputs may require editing the labels
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                # Get gradient norms
                l2_sum = 0

                # clip_grad_norm_(self.model.parameters(), 0.2)
                # TODO concatenate all vectors, flatten, take
                with torch.no_grad():
                    count = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # Compute the L2 norm of the gradient
                            l2_norm = torch.norm(param.grad)
                            l2_sum += l2_norm.item()
                            count += 1

                optimizer.step()

                if self.use_wandb:
                    wandb.log(
                        {
                            f"loss_task_idx_{self.task_index}": loss.item(),
                            f"grad_norm_task_idx_{self.task_index}": l2_sum,
                        }
                    )

            # evaluate model 
            test_acc, test_task_spec_acc, test_backward_transfer, forward_transfer = self.evaluate_task(
                test_dataloader=test_dataloader, golden_model_accs=golden_model_accs
            )

        if model_save_path is not None:
            # For now as models are small just saving entire things
            torch.save(self.model, model_save_path)

        return test_acc, test_task_spec_acc, test_backward_transfer, forward_transfer, metric_vals

    def create_task(
        self,
        target_labels: Set[int],
        memory_set_manager: MemorySetManager,
        active: bool = False,
    ) -> Task:
        """Generate a task with the given target labels.

        Args:
            target_labels: Set of labels that this task uses.
            memory_set_manager: The memory set manager to use to create memory set.
            active: Whether this task is active or not.
        Returns:
            Task with the given target labels.
        """
        train_index = torch.where(
            torch.tensor([y.item() in target_labels for y in self.train_y])
        )
        test_index = torch.where(
            torch.tensor([y.item() in target_labels for y in self.test_y])
        )

        train_x = self.train_x[train_index]
        train_y = self.train_y[train_index]
        test_x = self.test_x[test_index]
        test_y = self.test_y[test_index]
        task = Task(train_x, train_y, test_x, test_y, target_labels, memory_set_manager)
        task.active = active

        return task

    def _get_task_dataloaders(
        self, use_memory_set: bool, batch_size: int
    ) -> Tuple[List[DataLoader], DataLoader]:
        """Get dataloaders for all tasks <= task_index
        
        Args:
            use_memory_set: Whether to use the memory set for tasks < task_index.
            batch_size: Batch size to use for training.
        Returns:
            Tuple of list of train dataloaders (one for each task) single test dataloader.
        """

        running_tasks = [task for task in self.tasks if task.active]
        if len(running_tasks) < self.task_index + 1:
            print("WARNING: Current labels does not include all previous tasks")
        assert len(running_tasks) != 0 

        terminal_task = running_tasks[-1]
        memory_tasks = running_tasks[:-1]  # This could be empty

        # Identify the labels for the combined dataset
        current_labels: List[int] = sorted(list(self._get_current_labels()))
        # y labels need to be adjusted
        label_switcher = {old_label:new_label for old_label, new_label in zip(current_labels, range(len(current_labels)))}
        switcher = lambda x :label_switcher[x]

        # Adjust the batch size
        total_tasks = len(running_tasks)
        per_task_batch_size = batch_size // total_tasks
        if per_task_batch_size == 0:
            per_task_batch_size = 1

        train_dataloaders = []
        test_x = []
        test_y = []
        for task in memory_tasks:
            if use_memory_set:
                train_x = task.memory_x
                train_y = task.memory_y.apply_(switcher)
            else:
                train_x = task.train_x
                train_y = task.train_y.apply_(switcher)
            test_x.append(task.test_x)
            test_y.append(task.test_y.apply_(switcher))

            train_dataloader = DataLoader(
                TensorDataset(train_x, train_y),
                batch_size=per_task_batch_size,
                shuffle=True,
            )

            assert len(train_dataloader) != 0

            train_dataloaders.append(train_dataloader)

        # Add the terminal task
        train_dataloader = DataLoader(
            TensorDataset(terminal_task.train_x, terminal_task.train_y.apply_(switcher)),
            batch_size=per_task_batch_size,
            shuffle=True,
        )
        train_dataloaders.append(train_dataloader)
            
        test_x.append(terminal_task.test_x)
        test_y.append(terminal_task.test_y.apply_(switcher))
        combined_test_x = torch.cat(test_x)
        combined_test_y = torch.cat(test_y)
        test_dataloader = DataLoader(
            TensorDataset(combined_test_x, combined_test_y),
            batch_size=batch_size,
            shuffle=True,
        )

        return train_dataloaders, test_dataloader

    def _get_combined_task_dataloader(
        self, use_memory_set: bool, batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Collect the datasets of all tasks <= task_index and return it as a
        single combined dataloader.

        Args:
            use_memory_set: Whether to use the memory set for tasks < task_index.
            batch_size: Batch size to use for training.
        Returns:
            Tuple of train dataloader then test dataloader.
        """

        # Get tasks
        #running_tasks = self.tasks[: self.task_index + 1]
        running_tasks = [task for task in self.tasks if task.active]
        if len(running_tasks) < self.task_index + 1:
            print("WARNING: Current labels does not include all previous tasks")

        terminal_task = running_tasks[-1]
        memory_tasks = running_tasks[:-1]  # This could be empty

        # Create a dataset for all active tasks

        if use_memory_set:
            memory_x_attr = "memory_x"
            memory_y_attr = "memory_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"
        else:
            memory_x_attr = "train_x"
            memory_y_attr = "train_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"

        test_x_attr = "test_x"
        test_y_attr = "test_y"

        combined_train_x = torch.cat(
            [getattr(task, memory_x_attr) for task in memory_tasks]
            + [getattr(terminal_task, terminal_x_attr)]
        )
        combined_train_y = torch.cat(
            [getattr(task, memory_y_attr) for task in memory_tasks]
            + [getattr(terminal_task, terminal_y_attr)]
        )
        combined_test_x = torch.cat(
            [getattr(task, test_x_attr) for task in running_tasks]
        )
        combined_test_y = torch.cat(
            [getattr(task, test_y_attr) for task in running_tasks]
        )


        # Randomize the train dataset
        n = combined_train_x.shape[0]
        perm = torch.randperm(n)
        combined_train_x = combined_train_x[perm]
        combined_train_y = combined_train_y[perm]

        # Identify the labels for the combined dataset
        current_labels: List[int] = sorted(list(self._get_current_labels()))
        # y labels need to be adjusted
        label_switcher = {old_label:new_label for old_label, new_label in zip(current_labels, range(len(current_labels)))}
        switcher = lambda x :label_switcher[x]
        combined_train_y.apply_(switcher)
        combined_test_y.apply_(switcher)

        # Put into batches
        train_dataset = TensorDataset(combined_train_x, combined_train_y)
        test_dataset = TensorDataset(combined_test_x, combined_test_y)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def next_task(self) -> None:
        """Iterate to next task"""
        self.task_index += 1
        if self.task_index >= len(self.tasks):
            raise IndexError("No more tasks")
        self.tasks[self.task_index].active = True

    def deactivate_past_tasks(self) -> None:
        """Deactivate all tasks with index < task_index"""
        for task in self.tasks[: self.task_index]:
            task.active = False

    def _get_current_labels(self) -> Set[int]:

        #running_tasks = self.tasks[: self.task_index + 1]
        running_tasks = [task for task in self.tasks if task.active]
        if len(running_tasks) < self.task_index +1:
            print("WARNING: Current labels does not include all previous tasks")

        return set.union(*[task.task_labels for task in running_tasks])
    
    def _get_num_activate_tasks(self) -> int:
        """Get number of active tasks"""
        return sum([task.active for task in self.tasks])

    def calculate_backward_transfer(self) -> Float:
        T = self.task_index
        # TODO Complete

    def set_model(self, model) -> None:
        self.model = model


class Cifar100Manager(ContinualLearningManager, ABC):
    """ABC for Cifar100 Manager. Handles downloading dataset"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        Returns:
            Tuple of train_x, train_y, test_x, test_y
        """
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Download and load the training data
        trainset = torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the testing data
        testset = torchvision.datasets.CIFAR100(
            root=dataset_path, train=False, download=True, transform=transform
        )

        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=False)
        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=False)

        return train_x, train_y.long(), test_x, test_y.long()


class Cifar100ManagerSplit(Cifar100Manager):
    """Continual learning on the split Cifar100 task.

    This has 5 tasks, each with 2 labels. [[0-19], [20-39], [40-59], [60-79], [80-99]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for Cifar100"""

        # TODO Make this task init a function of an input config file
        tasks = []
        label_ranges = [set(range(i, i + 20)) for i in range(0, 100, 20)]
        for labels in label_ranges:
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks

class Cifar100Manager20Split(Cifar100Manager):
    """Continual learning on the split Cifar100 task.

    This has 20 tasks, each with 5 labels. [[0-4], [5-9], ... , [95-99]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for Cifar100"""

        # TODO Make this task init a function of an input config file
        tasks = []
        label_ranges = [set(range(i, i + 5)) for i in range(0, 100, 5)]
        for labels in label_ranges:
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks


class Cifar10Manager(ContinualLearningManager, ABC):
    """ABC for Cifar10 Manager. Handles dataset loading"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        """
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Download and load the training data
        trainset = torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the testing data
        testset = torchvision.datasets.CIFAR10(
            root=dataset_path, train=False, download=True, transform=transform
        )

        # Classes in CIFAR-10 for ref ( "plane", "car", "bird", "cat",
        #                  "deer", "dog", "frog", "horse", "ship", "truck",)

        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=False)
        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=False)

        return train_x, train_y.long(), test_x, test_y.long()


class Cifar10ManagerSplit(Cifar10Manager):
    """Continual learning on the classic split Cifar10 task.

    This has 5 tasks, each with 2 labels. [[0,1], [2,3], [4,5], [6,7], [8,9]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        tasks = []
        for i in range(5):
            labels = set([2 * i, 2 * i + 1])
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks


class Cifar10Full(Cifar10Manager):
    """
    Cifar10 but 1 task running all labels.
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        labels = set(range(10))
        task = self.create_task(labels, self.memory_set_manager, active=False)
        task.active = True
        tasks = [task]

        return tasks


class MnistManager(ContinualLearningManager, ABC):
    """ABC for Mnist Manager. Handles loading dataset"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
            transfer_metrics=transfer_metrics,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        Returns:
            Tuple of train_x, train_y, test_x, test_y
        """
        # Define a transform to normalize the data
        # transform = transforms.Compose(
        #    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        # )
        transform = transforms.Compose([transforms.ToTensor()])

        # Download and load the training data
        trainset = torchvision.datasets.MNIST(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the test data
        testset = torchvision.datasets.MNIST(
            root=dataset_path, train=False, download=True, transform=transform
        )

        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=True)
        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=True)

        return train_x, train_y.long(), test_x, test_y.long()


class MnistManager2Task(MnistManager):
    """Continual learning with 2 tasks for MNIST, 0-8 and 9."""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            transfer_metrics=transfer_metrics,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file

        # Set up tasks
        # Task 1 should just contain examples in the dataset with labels from 0-8
        labels = set(range(9))
        task_1 = self.create_task(labels, self.memory_set_manager, active=True)

        # Task 2 should contain examples in the dataset with label 9
        task_2 = self.create_task(set([9]), self.memory_set_manager, active=False)

        return [task_1, task_2]


class MnistManagerSplit(MnistManager):
    """Continual learning on the classic split MNIST task.

    This has 5 tasks, each with 2 labels. [[0,1], [2,3], [4,5], [6,7], [8,9]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
        transfer_metrics: List[str] = None,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            transfer_metrics=transfer_metrics,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        tasks = []
        for i in range(5):
            labels = set([2 * i, 2 * i + 1])
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks
