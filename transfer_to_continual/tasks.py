from jaxtyping import Float
from torch import Tensor
from typing import Set
from data import MemorySetManager


class Task:
    """Class storing all the data for a certain task in a continual learning setting.

    Every task contains some gold standard train_set, all the information you would want for that task,
    then some test_set, which is used to evaluate the model on that task, and a memory set, which is used
    when the task is not the current primary continual learning task, but instead is in the past.
    """

    def __init__(
        self,
        train_x: Float[Tensor, "n f"],
        train_y: Float[Tensor, "n 1"],
        test_x: Float[Tensor, "m f"],
        test_y: Float[Tensor, "m 1"],
        task_labels: Set[int],
        memory_set_manager: MemorySetManager,
    ) -> None:
        """
        Args:
            train_x: The training examples for this task.
            train_y: The training labels for this task.
            test_x: The test examples for this task.
            test_y: The test labels for this task.
            task_labels: Set of labels that this task uses.
            memory_set_manager: The memory set manager to use to create the memory set.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.task_labels = task_labels

        self.memory_x, self.memory_y = memory_set_manager.create_memory_set(
            train_x, train_y
        )
        self.task_labels = task_labels
        self.active = False
