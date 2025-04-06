
print("🟢 SCRIPT HAS STARTED")

import sys
import os
import warnings
import logging
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from absl import logging as absl_logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- Setup Environment ---

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ['TFDS_DATA_DIR'] = r"/data/newc6477/VAE/Single_Beat/AllPhysionet/"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
absl_logging.set_verbosity(absl_logging.ERROR)

try:
    print("📦 Importing Helper...")
    from src.utils.helper import Helper
except Exception as e:
    print(f"❌ Failed to import Helper: {e}")
    raise

# --- Configuration ---
PATH = "/users/newc6477/VAE/12_Lead_VECG/results/LeadI_AllPhysionet_Test/test_is_split1"
BASE = os.path.join(PATH, "I")
SAVE_DIR = os.path.join(PATH, "evaluation_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_hyperparams_from_json(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    return {
        "alpha": params["coefficients"]["alpha"],
        "beta": params["coefficients"]["beta"],
        "gamma": params["coefficients"]["gamma"],
        "latent_dimension": params["latent_dimension"],
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"]
    }

def evaluate_models(df_train, df_test, hyperparams_list, save_dir):
    results = []
    reports_path = os.path.join(save_dir, "classification_reports")
    os.makedirs(reports_path, exist_ok=True)

    for i in range(len(df_train)):
        print(f"Evaluating model {i+1}/{len(df_train)}")
        hparams = hyperparams_list[i]
        latent_dim = hparams["latent_dimension"]

        X_train = df_train[i].iloc[:, :latent_dim].values
        X_test = df_test[i].iloc[:, :latent_dim].values
        y_train = np.array(df_train[i]['diagnostic'].tolist(), dtype=int)
        y_test = np.array(df_test[i]['diagnostic'].tolist(), dtype=int)

        knn = KNeighborsClassifier()
        multi_knn = MultiOutputClassifier(knn, n_jobs=-1)
        param_grid = {"estimator__n_neighbors": [3, 5]}
        grid = GridSearchCV(multi_knn, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        overall_acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(y_test.shape[1])])
        with open(os.path.join(reports_path, f"model_{i}_report.txt"), "w") as f:
            f.write(report)

        results.append({
            "model_index": i,
            "best_k": grid.best_params_["estimator__n_neighbors"],
            "accuracy": overall_acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            **hparams
        })

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.drop(columns=["model_index"])
    summary_df.to_csv(os.path.join(save_dir, "summary_results.csv"), index=False)
    return summary_df

def main():
    print("🟢 SCRIPT HAS STARTED")

    # --- Load models and hyperparams ---
    print(f"📁 Listing folders in: {BASE}", flush=True)

    folders = [f for f in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, f))]
    models, hyperparams_list = [], []

    for folder in folders:
        model_path = os.path.join(BASE, folder, "model_best.keras")
        hyperparams_path = os.path.join(BASE, folder, "params.json")
        if os.path.exists(model_path) and os.path.exists(hyperparams_path):
            print(f"Loading {folder}",flush=True)
            model = tf.keras.models.load_model(model_path, compile=False)
            models.append(model)
            hyperparams_list.append(extract_hyperparams_from_json(hyperparams_path))

    # --- Load embeddings ---
    train_splits = ['split2', 'split3', 'split4', 'split5']
    test_splits = ['split1']
    dataset_config = {'name': ['physionet'], 'split': train_splits, 'shuffle_size': 1024, 'batch_size': 1024}
    dataset_test = {'name': ['physionet'], 'split': test_splits, 'shuffle_size': 1024, 'batch_size': 1024}

    df_train, _ = Helper.get_embeddings_multiple_model(models, datasets=dataset_config, lead='I')
    df_test, _ = Helper.get_embeddings_multiple_model(models, datasets=dataset_test, lead='I')

    # --- Evaluate ---
    results_df = evaluate_models(df_train, df_test, hyperparams_list, SAVE_DIR)
    print(results_df.sort_values(by="f1_macro", ascending=False))

if __name__ == "__main__":
    main()
