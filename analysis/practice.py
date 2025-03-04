import sys
import os

# Get the absolute path of the project root
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
print(PROJECT_DIR)
# Add the `src/` folder to Python's path
sys.path.append(PROJECT_DIR)


import sys
import os
import warnings
import logging
from absl import logging as absl_logging


os.environ['TFDS_DATA_DIR'] = r"C:\Users\Thomas Kaprielian\tensorflow_datasets"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
absl_logging.set_verbosity(absl_logging.ERROR)

from utils.helper import Helper
from src.evaluate.visualizations import Visualizations

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
from neurokit2.signal import signal_smooth

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import ipywidgets as widgets
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf

# The resolution for saving images
DPI = 300

# The source path of the experiments and models
PATH = r"C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VECG\results\run_5"

# Some operations take some time in computation.
# Therefore, the stored intermediate results can be used to skip the respective computation.
USE_PRECOMPUTED = True

# The source path of the experiments and models
PATH = r"C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VECG\results\run_5"
BASE = PATH  # Set the correct path to your BASE directory
models = []

# Get all lead folders in BASE directory
folders = [f for f in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, f))]

for lead in folders:
    lead_path = os.path.join(BASE, lead)
    
    # Find the first subfolder within the current lead folder
    subfolders = [sf for sf in os.listdir(lead_path) if os.path.isdir(os.path.join(lead_path, sf))]
    
    if subfolders:
        first_subfolder = subfolders[0]  # Assuming we want the first available subfolder
        model_path = os.path.join(lead_path, first_subfolder, 'model_best.keras')
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            models.append(model)
        else:
            print(f"Warning: Model file not found at {model_path}")
    else:
        print(f"Warning: No subfolders found in {lead_path}")

print(f"Loaded {len(models)} models successfully.")


dataset_physionet_train = {'name': ['physionet'], 'split': 'train', 'shuffle_size': 1024, 'batch_size': 1024}
dataset_physionet_validation = {'name': ['physionet'], 'split': 'validation', 'shuffle_size': 1024, 'batch_size': 1024}
dataset_physionet_test = {'name': ['physionet'], 'split': 'test', 'shuffle_size': 1024, 'batch_size': 1024}

df_physionet_train, ld = Helper.get_embeddings(models, dataset_physionet_train)
#df_physionet_validation, _ = Helper.get_embeddings(model, dataset_physionet_validation)
df_physionet_test, _ = Helper.get_embeddings(models, dataset_physionet_test)

df_physionet_train = df_physionet_train[0]
df_physionet_test = df_physionet_test[0]

import numpy as np

def one_hot_to_labels(y_onehot, class_labels):
    """
    Converts a one-hot encoded label matrix to human-readable class labels.
    
    Args:
        y_onehot (np.array): One-hot encoded labels (shape: num_samples × num_classes).
        class_labels (list): List of class names corresponding to each index in the one-hot encoding.

    Returns:
        np.array: Array of string labels, where multiple diagnoses are joined by "_".
    """
    y_labels = []
    for row in y_onehot:
        indices = np.where(row == 1)[0]  # Find indices of '1's in the row
        if len(indices) == 0:
            label = 0  # Assign to a new class
        else:
            label = "".join(class_labels[indices])  # Join multiple labels if multi-class

            print(int(label))
        y_labels.append(int(label))

    return np.array(y_labels)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print(type(df_physionet_train))
print(list(df_physionet_train.columns.values))
# Load your DataFrame (assuming it's named df)
X_train = df_physionet_train.iloc[:, :96].values  # Extract feature matrix (num_samples, 96)
X_test = df_physionet_test.iloc[:, :96].values
y_train = df_physionet_train['diagnostic'].values  # Extract one-hot encoded labels
y_test = df_physionet_test['diagnostic'].values
# Convert one-hot encoded labels to unique string labels
class_labels = np.array([
        '164889003', '164890007', '6374002', '426627000', '733534002',
        '713427006', '270492004', '713426002', '39732003', '445118002',
        '164947007', '251146004', '111975006', '698252002', '426783006',
        '284470004', '10370003', '365413008', '427172004', '164917005',
        '47665007', '427393009', '426177001', '427084000', '164934002',
        '59931005'
    ]) 
# Convert one-hot encoding to class labels

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
print("✅ Checking y_train format")
print("Type:", type(y_train))
print("Shape:", y_train.shape)
print("First few labels:", y_train[:5])
# Convert y_train to a proper NumPy array
y_train = np.array(y_train.tolist(), dtype=int)  # Convert list of lists to NumPy int array
y_test = np.array(y_test.tolist(), dtype=int)

# Confirm new structure
print("✅ Fixed y_train format")
print("New shape:", y_train.shape)  # Should be (num_samples, num_classes)
print("First row:", y_train[0])  # Should be a one-hot vector
knn = KNeighborsClassifier(n_neighbors=5)

# Wrap it with MultiOutputClassifier
multi_label_knn = MultiOutputClassifier(knn)

# Train the model
multi_label_knn.fit(X_train, y_train)  # Keep y_train as one-hot encoded

# Predict on test data
y_pred = multi_label_knn.predict(X_test)
