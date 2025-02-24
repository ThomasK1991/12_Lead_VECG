import sys
import os

# Get the absolute path of the project root
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# Add the `src/` folder to Python's path
sys.path.append(PROJECT_DIR)

# Now try importing the model
from model.tcvae import TCVAE
from model.encoder import Encoder
from model.decoder import Decoder

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

from src.utils.helper import Helper
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

datasets = {
    'name': ['physionet'],
    'split': 'train',
    'shuffle_size': 1024,
    'batch_size': 1024,
}

print('boo')
df = Helper.experiments(datasets, path=PATH)

