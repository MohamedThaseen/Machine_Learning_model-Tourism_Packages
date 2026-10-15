# Visit with Us: Wellness Tourism Package Prediction (MLOps Pipeline)

An end-to-end MLOps project for **"Visit with Us"**, a travel company launching a new **Wellness Tourism Package**. It predicts whether a customer will purchase the package *before* they are contacted, and automates the workflow (data registration, preparation, training, deployment) with **GitHub Actions** CI/CD, **MLflow** experiment tracking and **Hugging Face** hosting.

- GitHub repository: https://github.com/MohamedThaseen/Tourism_Packages
- Live app (Streamlit on Hugging Face Spaces): https://huggingface.co/spaces/Thaseen75/Tourism_Package_Prediction

## Table of Contents
- [Business Context](#business-context)
- [Objective](#objective)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Modelling](#modelling)
- [Deployment](#deployment)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Known Issues and Improvements](#known-issues-and-improvements)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

## Business Context
Identifying likely buyers of a new package by hand is inconsistent, slow and error-prone, which leads to missed opportunities and weak campaign performance. The company wants a scalable, automated system that integrates customer data, predicts likely buyers, and keeps the model up to date as customer behaviour changes.

## Objective
Design and deploy an MLOps pipeline that covers data cleaning, preprocessing, model building, training, evaluation and deployment, and that uses GitHub Actions so that a push to `main` rebuilds and redeploys the solution automatically.

## Dataset
`tourism.csv`, with the target **`ProdTaken`** (1 = purchased, 0 = did not purchase).

| Group | Features |
|-------|----------|
| Customer details | `Age`, `TypeofContact`, `CityTier`, `Occupation`, `Gender`, `NumberOfPersonVisiting`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `OwnCar`, `NumberOfChildrenVisiting`, `Designation`, `MonthlyIncome` |
| Interaction data | `PitchSatisfactionScore`, `ProductPitched`, `NumberOfFollowups`, `DurationOfPitch` |
| Identifier | `CustomerID` (dropped before modelling) |

> The raw data is registered as a Hugging Face **dataset** repository by the pipeline and is not stored in this repository.

## Pipeline Overview
```
push to main
   │
   ▼
register-dataset ─▶ data-prep ─▶ model-training ─▶ deploy-hosting
 (upload raw data     (clean, split,   (tune XGBoost,      (push Streamlit app
  to HF dataset)       upload splits)   log to MLflow,       to HF Space)
                                        upload model to HF)
```

| Step | Script | What it does |
|------|--------|--------------|
| Register data | `Model_building/data_register.py` | Creates the HF dataset repo if needed and uploads the raw data |
| Prepare data | `Model_building/prep.py` | Loads data from the Hub, cleans it, splits train/test and uploads `Xtrain`, `Xtest`, `ytrain`, `ytest` |
| Train | `Model_building/train.py` | Tunes and evaluates an XGBoost pipeline with MLflow tracking, then uploads the model to the HF model repo |
| Host | `hosting/hosting.py` | Creates the Streamlit Space if needed and uploads the `deployment/` folder |

**Data preparation details**
- Fixes inconsistent labels: `Fe Male` becomes `Female`, `Unmarried` becomes `Single`.
- Drops the index column, `CustomerID` and `CityTier`.
- 80/20 train/test split with `random_state=42`.

## Modelling
- **Algorithm**: `XGBClassifier` inside a scikit-learn pipeline.
- **Preprocessing**: `StandardScaler` on numeric features and `OneHotEncoder` (unknown categories ignored) on categorical features (`TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation`).
- **Class imbalance**: `scale_pos_weight` set from the class ratio and also searched over `[5, 10, 20, 50]`.
- **Tuning**: `RandomizedSearchCV`, 50 iterations, 5-fold CV, scored on accuracy. Search space covers `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` and `scale_pos_weight`.
- **Decision threshold**: 0.45 on predicted probability.
- **Tracking**: every parameter combination is logged as a nested MLflow run with its mean and standard deviation of CV score. The best parameters plus train and test accuracy, precision, recall and F1 are logged on the parent run, and the model file is stored as an artifact.
- **Output**: `Tourism_Package_Prediction_Model_v1.joblib`, uploaded to the Hugging Face model repo.

> Final train and test metrics are recorded in MLflow rather than in the notebook output, so add them here once you have them from your MLflow run:
>
> | Split | Accuracy | Precision | Recall | F1 |
> |-------|----------|-----------|--------|----|
> | Train | | | | |
> | Test | | | | |

## Deployment
A **Streamlit** app (`deployment/app.py`) downloads the model from the Hugging Face Hub and lets a user enter customer and pitch details to get a prediction of whether the customer will buy the package. A `Dockerfile` and `requirements.txt` are included in `deployment/`. Hosting is done on a Hugging Face Space by `hosting/hosting.py`.

## CI/CD with GitHub Actions
The workflow (`.github/workflows/pipeline.yml`) is triggered on every push to `main` and runs four dependent jobs on `ubuntu-latest` with Python 3.10: `register-dataset`, `data-prep`, `model-training` and `deploy-hosting`. The training job uploads the `mlruns` folder as a workflow artifact.

**Required repository secret**: `HF_TOURISM_PACKAGE_PREDICTION_TOKEN`, a Hugging Face access token with write permission (Settings → Secrets and variables → Actions).

## Known Issues and Improvements
- **Token variable mismatch in the workflow.** The `register-dataset`, `data-prep` and `deploy-hosting` jobs set `HF_TOKEN`, but `data_register.py`, `prep.py` and `hosting.py` read `HF_TOURISM_PACKAGE_PREDICTION_TOKEN`. Use the same variable name in the workflow and the scripts (only `model-training` currently matches).
- **Workflow file location.** In the notebook the YAML is stored in a markdown cell. It must be saved as `.github/workflows/<name>.yml` in the repository for GitHub Actions to run it.
- **Imbalanced target scored on accuracy.** For a purchase-prediction campaign, recall or F1 (or PR-AUC) is usually a better tuning metric than accuracy.
- **Unstratified split.** Use `stratify=y` so the buyer ratio is preserved in train and test.
- **Notebook training cell differs from `train.py`.** The notebook experiment keeps `CityTier` as a feature while the production script drops it. Keep them consistent.
- **Hosting.** The Dockerfile is included, but the Space is created with the Streamlit SDK, which ignores it. Either remove the Dockerfile or create a Docker Space.
- **Model repo creation.** `create_repo` is passed `space_sdk="streamlit"` for a model repository, where it has no effect.
- **Security.** See the warning below.

> ### Never commit credentials
> Access tokens (Hugging Face, GitHub, ngrok) must stay out of notebooks and source files. Use GitHub Actions secrets, environment variables or Colab secrets. If a token has ever been pasted into code or pushed anywhere, revoke it and create a new one.

## Getting Started

### Prerequisites
- Python 3.10+
- A Hugging Face account and write access token
- A GitHub repository with the secret above configured

### Install
```bash
git clone https://github.com/MohamedThaseen/Tourism_Packages.git
cd Tourism_Packages
pip install -r "Tourism Project/requirements.txt"
```

### Run the pipeline locally
```bash
export HF_TOURISM_PACKAGE_PREDICTION_TOKEN=<your-token>
python "Tourism Project/Model_building/data_register.py"
python "Tourism Project/Model_building/prep.py"
python "Tourism Project/Model_building/train.py"
python "Tourism Project/hosting/hosting.py"
```

### Run the app locally
```bash
cd "Tourism Project/deployment"
pip install -r requirements.txt
streamlit run app.py
```

### View MLflow runs
```bash
mlflow ui   # then open http://localhost:5000
```

## Repository Structure
```
.
├── .github/workflows/pipeline.yml       # CI/CD workflow
└── Tourism Project/
    ├── data/                             # raw data (tourism.csv)
    ├── Model_building/
    │   ├── data_register.py
    │   ├── prep.py
    │   └── train.py
    ├── deployment/
    │   ├── app.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── hosting/
    │   └── hosting.py
    └── requirements.txt
```

## Tech Stack
Python, pandas, scikit-learn, XGBoost, MLflow, Hugging Face Hub (datasets, models, Spaces), Streamlit, GitHub Actions.

## Author
Mohamed Rifakath Ali A
