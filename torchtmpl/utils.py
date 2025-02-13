import datetime
import functools
import logging
import os

import segmentation_models_pytorch as smp
import torch
import torch.nn
import tqdm
import yaml
from torch.amp import GradScaler, autocast

from . import models


def amp_autocast(func):
    """Décorateur pour exécuter une fonction avec autocast AMP."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.amp.autocast(device_type="cuda"):
            return func(*args, **kwargs)

    return wrapper


def generate_unique_logpath(logdir, raw_run_name):
    """Generate a unique directory name ensuring logs are saved in the home directory."""
    # Ensure logdir is stored in HOME while keeping it relative in YAML
    home_logdir = os.path.join(
        os.path.expanduser("~"), logdir
    )  # Expands ~/logs → /home/username/logs

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # AAAAMMJJ-HHMMSS
    i = 0

    while True:
        run_name = f"{raw_run_name}_{i}_{timestamp}"
        log_path = os.path.join(home_logdir, run_name)

        if not os.path.isdir(log_path):
            os.makedirs(log_path, exist_ok=True)
            return log_path
        i += 1


class ModelCheckpoint:
    """Early stopping callback."""

    def __init__(
        self,
        model: torch.nn.Module,
        savepath: str,
        min_is_best: bool = True,
    ) -> None:
        """Initialize a modelCheckpint instance.

        Args:
            model (torch.nn.Module): The model to monitor.
            savepath (str): The file path where to save the model
            min_is_best (bool, optional): Boolean to keep the lowest score. Defaults to True.

        """
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
    """Train a model for one epoch, iterating over the loader.

    This function iterates over the loader, computes the loss with f_loss, and updates the model parameters with the optimizer.

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
        inputs = inputs.to(device, memory_format=torch.channels_last)
        targets = targets.to(device)

        optimizer.zero_grad()  # Toujours remettre les gradients à zéro avant le passage avant

        # Forward pass avec précision mixte
        with autocast("cuda"):
            outputs = model(inputs).squeeze(1)
            loss = f_loss(outputs, targets)

        # Backward pass et optimisation
        scaler.scale(loss).backward()

        # Clipping des gradients pour éviter les explosions de gradients
        # scaler.unscale_(optimizer)  # Nécessaire avant clip_grad_norm_ avec AMP
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

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
        # if i % 10 ==0:
        pbar.set_description(f"Train loss : {total_loss / num_samples:.4f}")
    return total_loss / num_samples


@amp_autocast
def test(model, loader, f_loss, device):
    """Test a model over the loader using SMP metrics.

    Args:
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
            inputs = inputs.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
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
                preds, targets, mode="binary", threshold=0.5
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


def load_model_config(test_config):
    model_config_path = os.path.join(
        test_config["model_path"], test_config["model_config"]
    )
    logging.info(f"Loading model configuration from: {model_config_path}")
    with open(model_config_path) as f:
        return yaml.safe_load(f)["model"]


def build_and_load_model(
    model_config, input_size, num_classes, device, inference, model_path=None
):
    model = models.build_model(model_config, input_size, num_classes, inference)
    model.to(device, memory_format=torch.channels_last)

    if not inference:
        # Augmente la limite du cache de compilation pour éviter des recompilations
        torch._dynamo.config.cache_size_limit = 512  
        torch._dynamo.config.suppress_errors = True  # n'applique pas la compilation sur les formes problématiques

        # Réduit la pression mémoire en limitant l'usage des CUDAGraphs et en ajustant le tuning
        torch._inductor.config.triton.cudagraphs = False  # Désactive CUDAGraphs pour éviter les problèmes de mémoire
        torch._inductor.config.coordinate_descent_tuning = False  # Active l'optimisation mémoire

        # Ajuste la précision des multiplications matricielles
        torch.set_float32_matmul_precision('high')  

        # Fixe une limite stricte de mémoire partagée pour éviter les erreurs d’overflow
        import os
        os.environ["TORCHINDUCTOR_MAX_SHARED_MEMORY"] = "101376"  # Limite mémoire partagée

        # Compile avec les meilleurs paramètres pour éviter les dsépassements mémoire
        model = torch.compile(
            model,
            backend="inductor",
            mode="default",   # <- Évite le tuning trop massif
            dynamic=True,             # <- Autorise différentes formes de batch
            fullgraph=False  # Désactive fullgraph pour éviter les erreurs liées aux recompilations
        )



    if inference:
        if model_path is None:
            raise ValueError("model_path must be provided in inference mode.")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

    return model


def apply_tta(model, use_tta):
    """Wrap the model with TTA if use_tta is True."""
    import ttach as tta

    if use_tta:
        logging.info("Test-Time Augmentation (TTA) ACTIVÉ")
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
        return tta.SegmentationTTAWrapper(model, tta_transforms)
    else:
        logging.info("Test-Time Augmentation (TTA) DÉSACTIVÉ")
        return model
