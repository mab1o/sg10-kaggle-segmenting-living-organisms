# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    scaler = GradScaler("cuda")  # Initialiser le scaler pour la précision mixte

    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Toujours remettre les gradients à zéro avant le passage avant

        # Forward pass avec précision mixte
        with autocast("cuda"):
            outputs = model(inputs).squeeze(1)
            loss = f_loss(outputs, targets)

        # Backward pass et optimisation
        scaler.scale(loss).backward()

        # Clipping des gradients pour éviter les explosions de gradients
        #scaler.unscale_(optimizer)  # Nécessaire avant clip_grad_norm_ avec AMP
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)


        scaler.step(optimizer)
        scaler.update()

        # Vidage de mémoire et diagnostic (à espacer pour éviter des ralentissements)
        # if i % 300 == 0 and device == torch.device("cuda"):
        #   print("Avant vidage du cache:")
        #   print(torch.cuda.memory_summary())
        #   torch.cuda.empty_cache()
        #   print("Après vidage du cache:")
        #  print(torch.cuda.memory_summary())

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.4f}")
    return total_loss / num_samples


def test(model, loader, f_loss, device):
    """
    Test a model over the loader using SMP metrics.

    Arguments:
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- A torch.device

    Returns:
        avg_loss      -- The average loss over the dataset
        avg_f1        -- The F1-score computed over the dataset
        avg_precision -- The precision computed over the dataset
        avg_recall    -- The recall computed over the dataset
    """
    
    model.eval()
    total_loss, num_samples = 0, 0

    # Initialize true positive, false positive, false negative, true negative
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze(1)
            loss = f_loss(outputs, targets)

            # Compute loss sum
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

            # Convert logits to binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).int()
            
            targets = targets.int()  # Ajoute cette ligne

            # Get stats for precision, recall, F1 computation
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                preds, targets, mode='binary', threshold=0.5
            )

            # Aggregate stats
            tp += batch_tp.sum()
            fp += batch_fp.sum()
            fn += batch_fn.sum()
            tn += batch_tn.sum()

    # Compute final metrics
    avg_loss = total_loss / num_samples
    avg_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    avg_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
    avg_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

    return avg_loss, avg_f1, avg_precision, avg_recall