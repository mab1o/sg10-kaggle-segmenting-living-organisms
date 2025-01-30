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
from . import losses

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

    # Build the loss
    logging.info("= Loss")
    loss_config = config["loss"]
    if loss_config["name"] == "BCEWithLogitsLoss":
        pos_weight = torch.tensor([loss_config.get("pos_weight", 1.0)], device=device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_config["name"] == "TverskyLoss":
        loss = losses.TverskyLoss(alpha=loss_config.get("alpha", 0.3), beta=loss_config.get("beta", 0.7))
        print(f"Using TverskyLoss with alpha={loss_config['alpha']} and beta={loss_config['beta']}")
    elif loss_config["name"] == "FocalTverskyLoss":
        loss = losses.FocalTverskyLoss(
            alpha=loss_config.get("alpha", 0.3),
            beta=loss_config.get("beta", 0.7),
            gamma=loss_config.get("gamma", 1.1)
        )
        print(f"Using FocalTverskyLoss with alpha={loss_config['alpha']}, beta={loss_config['beta']}, and gamma={loss_config['gamma']}")
    else:
        loss = optim.get_loss(loss_config["name"])

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config["optim"], model.parameters())

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

        # Test
        test_loss, test_f1 = utils.test(model, valid_loader, loss, device)

        # Mise à jour du checkpoint si meilleur F1-score
        updated = model_checkpoint.update(test_f1)
        logging.info(
            "[%d/%d] Test loss : %.4f, Test F1-score : %.4f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                test_f1,
                "[>> BETTER F1-score <<]" if updated else "",
            )
        )

        # Log dans wandb
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log({"train_CE": train_loss, "test_CE": test_loss, "test_F1": test_f1})




# Use for visualize and validate result with binary prediction.
def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model_name = config['test']['model_path'] + config['test']['model_name']
    config = yaml.safe_load(open(config['test']['model_path']+ config['test']['model_config'], "r"))

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config['data']['trainpath'],config['data']['patch_size'],mode='test')
    #dataset_teset = config .. trainpath ??
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


    # Seconde partie de test: Utiliser les prédiction
    logging.info("= Predict first image")
    for idx_img in range(dataset_test.image_patches[0][0]*dataset_test.image_patches[0][1]):
        if(idx_img % 400 == 0):
            logging.info(f"  - predict mask {idx_img}")
        image = dataset_test[idx_img].unsqueeze(0).to(device)
        dataset_test.insert(model.predict(image))
    
    print(dataset_test.mask_files[0][0])

    logging.info("= Reconstruct image")
    #dataset_test = alors qu'on veut comparer mask réel et image ??
    data.show_image_mask_from(dataset_test,0,"image_reconstruct_1.png")

    logging.info("= Compare masks")
    #data.show_mask_predict_compare_to_real(dataset_test,0,dataset_train,"compare_mask_1.png")



# Use for visualize and validate result with probability instead of binary prediction.
def test_proba(config):
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
        probs = model.predict_probs(image).half()
        dataset_train_proba.insert(probs)  # Stocke uniquement les probabilités dans ce dataset

    # Visualiser le mask de proba prédit vs le mask binaire réel
    # et calcule le meilleur seuil pour obtenir le plus grand F1-score
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
    model.load_state_dict(torch.load(model_name))
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
            
            # put threshold at the value given py test_proba.
            threshold = 0.5
            dataset_test.insert(model.predict(image, threshold))


    logging.info("= To submit")
    dataset_test.to_submission()




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
    valid_commands = ["train", "test", "test_proba", "sub"]
    if command not in valid_commands:
        logging.error(f"Invalid command: {command}. Valid commands are: {', '.join(valid_commands)}")
        sys.exit(-1)

    # Execute the command
    eval(f"{command}(config)")