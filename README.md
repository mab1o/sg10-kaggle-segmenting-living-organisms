# Kaggle - Segmenting living organisms

## About

This project was developed as part of a [Kaggle challenge](https://www.kaggle.com/competitions/3md4040-2025-challenge) for the 3rd-year Deep Learning course. The objective is to perform pixel-level detection of living organisms—specifically plankton—in large grayscale images of approximately 22,000 x 14,000 resolution. The challenge involves designing a robust deep learning solution capable of accurately identifying plankton at this fine granularity, making it a complex problem at the intersection of computer vision and scientific research. The project not only tackles the technical challenges of high-resolution image processing but also contributes to advancing knowledge in biological imaging and automated plankton detection.

## Usage

### Getting Started

1. Setting up the environment :

    ```bash
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    python -m pip install .
    ```

2. Run a training :

    ```bash
    python -m torchtmpl.main config.yaml train
    ```

3. Create the submission.csv

    ```bash
    python -m torchtmpl.main config.yaml sub
    ```

4. Submit the submission.csv file:

    ```bash
    kaggle competitions submit -c 3md4040-2025-challenge -f submission.csv -m "Message"
    ```

5. Visualize the data and gain insights

```bash
python -m torchtmpl.main config.yaml test
```
```bash
python -m torchtmpl.main config.yaml test_proba
```


### Cluster experimentation

For running the code on a cluster, we provide an example script for starting an experimentation on a SLURM based cluster.
The script we provide is dedicated to a use on our clusters and you may need to adapt it to your setting. 
Then running the simulation can be as simple as :

1. Connect to the frontal with SSH [🔗 Guide](https://dce.pages.centralesupelec.fr/03_connection/#using-visual-studio-code)  
2. Commit your changes  
3. Submit the job:
```bash
python3 submit-slurm.py {yaml_name}.yaml
```


### Using the Ensemble Method

#### **1. Ensure you have trained multiple models**
Each trained model should be stored inside the `ensemble_models/` directory with the following structure:

ensemble_models/
│── Model_1/
│   ├── best_model.pt
│   ├── config.yaml
│
│── Model_2/
│   ├── best_model.pt
│   ├── config.yaml
│
│── Model_3/
│   ├── best_model.pt
│   ├── config.yaml

#### **2. Run ensemble inference**
```bash
python3 -m torchtmpl.main ensemble.yaml sub_ensemble
```

#### **3. Submit the submission.csv file:**
```bash
kaggle competitions submit -c 3md4040-2025-challenge -f submission.csv -m "Message"
```


## Project Structure

### Tree

```plaintext
├── config.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
├── submit-slurm.py
└── torchtmpl
    ├── data         # data management
    ├── __init__.py
    ├── main.py
    ├── models       # model management
    ├── optim.py
    └── utils.py
```

### Projet Choices

To have more details on the project decisions go see : [the Wiki](https://gitlab-student.centralesupelec.fr/margaux.blondel/kaggle-segmenting-living-organisms/-/wikis/home).

## Minimum Viable Product (MVP) milestones

- [x] **DATA** : write a PlanktonDataset as pytorch Dataset
- [x] **MODEL** : implement a basic baseline model,
- [x] **METRICS** : implement the basic metrics of interest, in particulier F1 score
- [x] **SUBMISSION** : Write the test function to generate the submission file
- [x] **LOG** : Consider logging your experiments to online dashboards (e.g. wandb)
- [x] **DCE** : Run your first experiments with basic models, basic pipelines, small set of training data, and make your first submissions

## Useful links

- [Regression Problem Example](https://frezza.pages.centralesupelec.fr/teachml2/Supports/NeuralNetworks/02-ffn.html#/example-on-a-regression-problem-1)
- [Evaluated criterion](https://frezza.pages.centralesupelec.fr/teachml2/Supports/NeuralNetworks/00-intro.html#/evaluation-33/0)
