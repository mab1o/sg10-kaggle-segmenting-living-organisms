# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

# Local imports
from . import data
from . import models
from . import optim
from . import utils
from torchtmpl.data.transformations import get_transforms

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Initialisation de wandb si activé
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
        wandb_log = wandb.log
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Charger le modèle pré-entraîné si un chemin est spécifié
    if "pretrained_model" in config and os.path.exists(config["pretrained_model"]):
        logging.info(f"Loading pretrained model from {config['pretrained_model']}")
        model.load_state_dict(torch.load(config["pretrained_model"], map_location=device, weights_only=True))
    #else:
        #logging.warning("No pretrained model found, training from scratch.")


    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"]["name"], config["loss"], device)

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config["optim"], model.parameters())

    # Build the scheduler
    logging.info("= Scheduler")
    scheduler = optim.get_scheduler(optimizer, config["scheduler"])



    # Build the logging directory
    logdir = pathlib.Path(utils.generate_unique_logpath(config["logging"]["logdir"], model_config["class"]))
    logdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Will be logging into {logdir}")

    # Sauvegarde locale du fichier config.yaml
    config_path = logdir / "config.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Ajout du config.yaml en tant qu'Artifact dans WandB (uniquement si wandb est activé)
    if wandb_log is not None:
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(str(config_path))
        wandb.log_artifact(artifact)


    chosen_transforms = get_transforms(config["data"]["transform_type"])
    logging.info(f'Niveau de transformation: {(config["data"]["transform_type"])}')
    # Afficher dans le terminal avec logging
    logging.info(f"Transformations appliquées : {chosen_transforms}")

    # Make a summary script of the experiment
    logging.info("= Summary")
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n" + " ".join(sys.argv) + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=next(iter(train_loader))[0].shape)}\n\n"
        + "## Loss\n\n" + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    
    # Écriture du résumé de l'expérience
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Initialisation du checkpointing
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )
    
    logging.info("= Start training")
    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        test_loss, test_f1, test_precision, test_recall = utils.test(model, valid_loader, loss, device)

        scheduler.step()  # fonctionnement du scheduler.

        # Mise à jour du checkpoint si meilleur F1-score
        updated = model_checkpoint.update(test_f1)
        logging.info(
            "[%d/%d] Test loss : %.4f, Precision : %.4f, Recall : %.4f, Test F1-score : %.4f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                test_precision,
                test_recall,
                test_f1,
                "[>> BETTER F1 <<]" if updated else "",
            )
        )

        # Log dans wandb
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log({
                "train_CE": train_loss,
                "test_CE": test_loss,
                "test_Precision": test_precision,
                "test_Recall": test_recall,
                "test_F1": test_f1
            })

def test(config):
    """Use for visualize and validate result with binary prediction"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model_name = config['test']['model_path'] + config['test']['model_name']
    config = yaml.safe_load(open(config['test']['model_path']+ config['test']['model_config'], "r"))

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config['data']['testpath'],config['data']['patch_size'],mode='test')
    dataset_train = data.PlanktonDataset(
        config['data']['trainpath'],config['data']['patch_size'])
    input_size = tuple(dataset_train[0][0].shape)
    print ("="*90, input_size)
    num_classes = input_size[0]

    print(len(dataset_test))
    print(len(dataset_train))

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()

    # Seconde partie de test: Utiliser les prédiction
    logging.info("= Predict first image")
    for idx_img in range(dataset_test.image_patches[0][0]*dataset_test.image_patches[0][1]):
        if(idx_img % 400 == 0):
            logging.info(f"  - predict mask {idx_img}")
        image = dataset_test[idx_img].unsqueeze(0).to(device)
        dataset_test.insert(model.predict(image))
    
    print(dataset_test.mask_files[0][0])

    #logging.info("= Reconstruct image")
    #dataset_test = alors qu'on veut comparer mask réel et image ??
    #data.show_image_mask_from(dataset_train,0,"image_reconstruct_1.png")

    #logging.info("= Compare masks")
    #data.show_mask_predict_compare_to_real(dataset_test,0,dataset_train,"compare_mask_1.png")

    logging.info("= Compare validation image with predicted mask")
    data.show_validation_image_vs_predicted_mask(
        ds=dataset_test,            # Dataset contenant les masques prédits
        idx=0,                      # Index de l'image à afficher
        validation_dataset=dataset_test,  # Dataset avec l'image de validation
        image_name="validation_vs_predicted_1.png"
    )

def test_proba(config):
    """Use for visualize and validate result with probability instead of binary prediction"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model_name = config['test']['model_path'] + config['test']['model_name']
    config = yaml.safe_load(open(config['test']['model_path']+ config['test']['model_config'], "r"))

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config['data']['testpath'],config['data']['patch_size'],mode='test')
    dataset_train = data.PlanktonDataset(
        config['data']['trainpath'],config['data']['patch_size'])
    input_size = tuple(dataset_train[0][0].shape)
    print ("="*90, input_size)
    num_classes = input_size[0]

    print(len(dataset_test))
    print(len(dataset_train))

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Seconde partie de test_with_proba: mask de proba prédit vs le mask binaire réel
    # or seul dataset_train à des masks binaire réels.
    logging.info("= Predict probabilities and compare to real masks")
    dataset_train_proba = data.PlanktonDataset(  # Copie dédiée aux probabilités
        config['data']['trainpath'], config['data']['patch_size'],
        mode='test'  # autorise dataset_train_proba.insert(...)

    )
    for idx_img in range(dataset_train.image_patches[0][0] * dataset_train.image_patches[0][1]):
        if idx_img % 400 == 0:
            logging.info(f"  - Predicting probabilities for mask {idx_img}")
        image = dataset_train[idx_img][0].unsqueeze(0).to(device)

        # Récupérer les probabilités
        with torch.amp.autocast(device_type='cuda'), torch.inference_mode():
            if 'segmentation_models_pytorch' in type(model).__module__:
                # Cas segmentation_models_pytorch
                logits = model(image)
                probs = torch.sigmoid(logits).half()  # Convertit les logits en probabilités [0,1]
            else:
                probs = model.predict_probs(image).half()
            
        dataset_train_proba.insert(probs)  # Stocke uniquement les probabilités dans ce dataset

    # Visualiser le mask de proba prédit vs le mask binaire réel
    # et calcule le meilleur seuil pour obtenir le plus grand F1-score
    logging.info("= Show probabilities and compare to real masks")
    data.show_predicted_mask_proba_vs_real_mask_binary(dataset_train_proba, 0, dataset_train, "proba_compared_real_1.png")


def sub(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model_name = config['test']['model_path'] + config['test']['model_name']
    config = yaml.safe_load(open(config['test']['model_path']+ config['test']['model_config'], "r"))

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config['data']['testpath'],config['data']['patch_size'],mode='test')
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]
    logging.info(f"  -  number of sample: {len(dataset_test)}")

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()

    logging.info("= Predict masks for all test images")
    for image_idx, (num_patches_x, num_patches_y) in enumerate(dataset_test.image_patches):
        logging.info(f"Predicting patches for image {image_idx}")
        for idx_patch in range(num_patches_x * num_patches_y):
            global_idx = idx_patch + sum(
                x * y for x, y in dataset_test.image_patches[:image_idx]
            )  # Calcul de l'index global
            #logging.info(f"  - predict mask {global_idx} (patch {idx_patch} in image {image_idx})")
            image = dataset_test[global_idx].unsqueeze(0).to(device)
            
            prediction = model.predict(image)

            # Apply sigmoid + threshold only for RegNetY models because it ouputs logits
            threshold = 0.5
            if config['model']['encoder']['model_name'] == "timm-regnety_032":
                prediction = (torch.sigmoid(prediction) > threshold).long()

            dataset_test.insert(prediction)

    logging.info("= To submit")
    dataset_test.to_submission()





def sub_ensemble(config):
    """Effectue une prédiction par ensemble de modèles"""
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config['data']['testpath'], config['data']['patch_size'], mode='test')
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]
    logging.info(f"  -  Number of samples: {len(dataset_test)}")

    # **Trouver tous les modèles dans `ensemble_models/`**
    ensemble_dir = "ensemble_models/"
    model_dirs = [os.path.join(ensemble_dir, d) for d in os.listdir(ensemble_dir) if os.path.isdir(os.path.join(ensemble_dir, d))]
    model_paths = [os.path.join(d, "best_model.pt") for d in model_dirs if os.path.exists(os.path.join(d, "best_model.pt"))]
    
    logging.info(f"Found {len(model_paths)} models in {ensemble_dir}: {model_paths}")

    if not model_paths:
        logging.error("No models found for ensembling! Check your ensemble_models/ directory.")
        return

    # **Chargement des modèles**
    logging.info("= Loading Models")
    models_list = []
    
    for model_path in model_paths:
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        logging.info(f"Loading model: {model_path} with config: {config_path}")

        if not os.path.exists(config_path):
            logging.error(f"⚠️ Missing config.yaml for {model_path}, skipping...")
            continue

        # Charger la configuration du modèle
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)["model"]

        # Construire et charger le modèle
        model = models.build_model(model_config, input_size, num_classes)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models_list.append(model)

    if not models_list:
        logging.error("No valid models could be loaded! Aborting ensemble prediction.")
        return

    logging.info(f"Loaded {len(models_list)} models successfully!")

    # **Prédictions pour toutes les images**
    logging.info("= Predict masks for all test images using ensemble")

    for image_idx, (num_patches_x, num_patches_y) in enumerate(dataset_test.image_patches):
        logging.info(f"Predicting patches for image {image_idx}")

        for idx_patch in range(num_patches_x * num_patches_y):
            global_idx = idx_patch + sum(
                x * y for x, y in dataset_test.image_patches[:image_idx]
            )  # Calcul de l'index global

            image = dataset_test[global_idx].unsqueeze(0).to(device)

            # Moyenne des prédictions de tous les modèles
            predictions = [torch.sigmoid(model(image)) for model in models_list]
            avg_prediction = torch.mean(torch.stack(predictions), dim=0)

            # Appliquer un seuil pour la segmentation binaire
            binary_prediction = (avg_prediction > 0.5).long()
            dataset_test.insert(binary_prediction)

    # **Sauvegarde de la soumission**
    logging.info("= Saving ensemble submission")
    dataset_test.to_submission()
    logging.info("Ensemble submission saved successfully!")





if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # check in advance if cuda is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available.")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test|test_proba|sub>")
        sys.exit(-1)

    logging.info(f"Loading configuration from {sys.argv[1]}")
    config = yaml.safe_load(open(sys.argv[1], "r"))

    # Ensure the command is valid
    command = sys.argv[2]
    valid_commands = ["train", "test", "test_proba", "sub", "sub_ensemble"]
    if command not in valid_commands:
        logging.error(f"Invalid command: {command}. Valid commands are: {', '.join(valid_commands)}")
        sys.exit(-1)

    # Execute the command
    eval(f"{command}(config)")
