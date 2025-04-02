#LD_LIBRARY_PATH=/usr/lib64 ssh headnode1

import argparse
import datetime
import os
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop
from utils.callbacks import ReconstructionPlot, CoefficientScheduler, CollapseCallback
from utils.helper import Helper
import random
from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Define the target directory
custom_path = r"/users/newc6477/VAE/12_Lead_VECG/src"

# Add to sys.path
if custom_path not in sys.path:
    sys.path.append(custom_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# TODO: Set path to the location of the tensorflow datasets
os.environ['TFDS_DATA_DIR'] = r"/data/newc6477/VAE/Single_Beat/5_percent_Physionet/"


class CustomModelCheckpoint(ModelCheckpoint):
    def _save_model(self, epoch, logs, batch=None):
        # If saving to a .keras file, force the SavedModel format.
        if self.filepath.endswith('.keras'):
            try:
                # Use save_format="tf" so that subclassed models can be saved.
                self.model.save(
                    self.filepath,
                )
            except Exception as e:
                print("Error saving model in CustomModelCheckpoint:", e)
        else:
            
            super()._save_model(epoch, logs, batch=batch)
def main(parameters,lead,i):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] + '/' + f'test_is_split{i}'+'/' +lead + '/'  + start_time + '/' 
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
    ## Ourput from load_multiple_data_sets is a list of tfds that are batched. Each member of the list corresponds to one of the datasets in paramters['train_dataset'] 
    ## Each member of the list is a batched tfds.datasets.datasets with all the leads.


    ######################################################
    # MACHINE LEARNING
    ######################################################
    callbacks = [
        TerminateOnNaN(),
        CollapseCallback(val, base_path + 'training/collapse/'),
        CSVLogger(base_path + 'training/training_progress.csv'),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping']), #training will stop if val_loss doesn't improve for patience number of epochs. For example:
        CoefficientScheduler(parameters['epochs'], parameters['coefficients'], parameters['coefficients_raise']),
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=20, min_lr=0.000001),
        CustomModelCheckpoint(filepath=base_path + 'model_best.keras', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train[0], base_path + 'training/reconstruction/', parameters['reconstruction'],lead),
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
        # Load training log
    log_path = base_path + 'training/training_progress.csv'
    df = pd.read_csv(log_path)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plot_path = base_path + 'training/loss_curve.png'
    plt.savefig(plot_path)

    csv_path = base_path
    save_path = os.path.join(csv_path,'kl_loss')
    csv_path =os.path.join(csv_path,'training_progress.csv')
    plt.close()
    plt.figure(figsize=(12, 8))
    plt.plot(df['epoch'], df['kl_loss'], label='Total KL Loss')
    plt.plot(df['epoch'], df['mi'], label='Mutual Information (MI)')
    plt.plot(df['epoch'], df['tc'], label='Total Correlation (TC)')
    plt.plot(df['epoch'], df['dw_kl'], label='Dimension-wise KL')
    plt.plot(df['epoch'], df['recon'], label='Reconstruction Loss', linestyle='--', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss Component Value')
    plt.title('KL Divergence Components and Reconstruction Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

    print(f"📉 Loss curve saved at: {plot_path}")

    # 🚀 2️⃣ Try Saving the Model Properly
    model_path = base_path + "model_final.keras"

    
    try:
        vae.save(model_path)
        print(f"\n✅ Model saved successfully at: {model_path}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")



if __name__ == '__main__':
    print("🧠 GPUs visible to TF:", tf.config.list_physical_devices('GPU'))
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
        {'latent_dimension': 20, 'coefficients': {'alpha': 6.01, 'beta': 0.3, 'gamma': 0.2}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 2.01, 'beta': 0.3, 'gamma': 0.2}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 2.01, 'beta': 0.3, 'gamma': 0.2}},  # current best
        {'latent_dimension': 18, 'coefficients': {'alpha': 2.01, 'beta': 0.3, 'gamma': 0.2}},
        {'latent_dimension': 22, 'coefficients': {'alpha': 2.01, 'beta': 0.3, 'gamma': 0.2}},
    ]


    

    twelve_leads = (['V4', 'V5', 'V6'])
    for i in range(1,2):
        all_splits = [f"split{j}" for j in range(1,6)]
        # Assign test split
        test_split = f"split{i}"
                # Assign training splits (exclude test)
        non_test_splits = [s for s in all_splits if s != test_split]
        val_split = non_test_splits[0]
        train_splits = [s for s in non_test_splits if s != val_split]

        parameters['train_dataset'] = {
            'name': ['physionet'],
            'split': train_splits,
            'shuffle_size': 1024,
            'batch_size': 1024,
        }

        parameters['val_dataset'] = {
            'name': ['physionet'],
            'split': val_split,
            'shuffle_size': 1024,
            'batch_size': 1024,
        }

        for lead in twelve_leads:   #Loops through each lead to train encoder and decoder

            for k in combinations: #Loops through the combinations of hyper parameters for that lead
                parameters.update(k)
                main(parameters,lead,i)
