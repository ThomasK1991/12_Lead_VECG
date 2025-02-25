import argparse
import datetime
import os

import tensorflow as tf
import keras
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop

from utils.callbacks import ReconstructionPlot, CoefficientScheduler, CollapseCallback
from utils.helper import Helper
import json 
from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE
import sys
# Define the target directory
custom_path = r"C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VECG\src"

# Add to sys.path
if custom_path not in sys.path:
    sys.path.append(custom_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# TODO: Set path to the location of the tensorflow datasets
os.environ['TFDS_DATA_DIR'] = r"C:\Users\Thomas Kaprielian\tensorflow_datasets"


def main(parameters,lead):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] +'/' +lead + '/'  + start_time + '/' 
    Helper.generate_paths(
        [base_path,base_path + 'training/reconstruction/', base_path + 'training/collapse/']
    )
    Helper.write_json_file(parameters, base_path + 'params.json')
    Helper.print_available_gpu()
    ######################################################
    # DATA LOADING
    ######################################################
    train, size_train = Helper.load_multiple_datasets(parameters['train_dataset'])
    print("/n Train Type" + str(type(train)))
    val, size_val = Helper.load_multiple_datasets(parameters['val_dataset'])

    ######################################################
    # MACHINE LEARNING
    ######################################################
    callbacks = [
        TerminateOnNaN(),
        CollapseCallback(val, base_path + 'training/collapse/'),
        CSVLogger(base_path + 'training/training_progress.csv'),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping']),
        CoefficientScheduler(parameters['epochs'], parameters['coefficients'], parameters['coefficients_raise']),
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=20, min_lr=0.000001),
        ModelCheckpoint(filepath=base_path + 'model_best.keras', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train[0], base_path + 'training/reconstruction/', parameters['reconstruction']),
    ]

    encoder = Encoder(parameters['latent_dimension'])
    decoder = Decoder(parameters['latent_dimension'])
    vae = TCVAE(encoder, decoder, parameters['coefficients'], size_train)
    vae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
    vae.fit(
        Helper.data_generator(train,lead=lead), steps_per_epoch=size_train,
        validation_data=Helper.data_generator(val,lead=lead), validation_steps=size_val,
        epochs=parameters['epochs'], callbacks=callbacks, verbose=1,
    )

    # 🚀 2️⃣ Try Saving the Model Properly
    model_path = base_path + "model_final.keras"

    
    try:
        vae.save(model_path)
        print(f"\n✅ Model saved successfully at: {model_path}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='VECG', description='Representational Learning of ECG using disentangling VAE',
    )
    parser.add_argument(
        '-p', '--path_config', type=str, default='./src/params.yml',
        help='location of the params file (default: ./params.yml)',
    )

    print("Current Directory:", os.getcwd())

    args = parser.parse_args()
    parameters = Helper.load_yaml_file(args.path_config)

    combinations = [
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
    ]

    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

    for lead in twelve_leads:   #Loops through each lead to train encoder and decoder

        for k in combinations: #Loops through the combinations of hyper parameters for that lead
            parameters.update(k)
            main(parameters,lead)
