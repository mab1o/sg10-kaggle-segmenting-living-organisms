# External imports
import torch
import torch.nn
import logging

from . import train_continue_utils


class ModelCheckpoint:
    """Callback class to save the model whenever a better score is achieved.

    Args:
        model (torch.nn.Module): The model to monitor.
        savepath_model (str): The file path to save the model.
        min_is_best (bool): If True, the model with the lowest score is considered the best.
                            If False, the model with the highest score is considered the best.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        savepath_model: str,
        savepath_checkpoint: str,
        min_is_best: bool = True,
        best_score: float = None,
    ) -> None:
        """Initialize a modelCheckpint instance.

        Args:
            model (torch.nn.Module): The model to monitor.
            savepath_model (str): The file path where to save the model
            min_is_best (bool, optional): Boolean to keep the lowest score. Defaults to True.

        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.savepath_model = savepath_model
        self.savepath_checkpoint = savepath_checkpoint
        self.best_score = best_score
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score, epoch):
        if self.is_better(score):
            logging.debug(f"Best_score is now {score}, old score is {self.best_score}")
            torch.save(self.model.state_dict(), self.savepath_model)
            train_continue_utils.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                score,
                self.savepath_checkpoint,
            )
            self.best_score = score
            return True
        logging.debug(f"Best_score is still {self.best_score}")
        return False
