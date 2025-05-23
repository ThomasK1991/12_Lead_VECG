import sys
import os

# Move up two levels to the root directory (where 'src' is a subfolder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import tensorflow as tf
from tensorflow import keras
import os, yaml, codecs, json
import numpy as np
import pandas as pd
import glob
import ipywidgets as widgets
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.metrics.disentanglement import Disentanglement
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
# Try importing normally (for scripts)
try:
    from model.encoder import Encoder
    from model.decoder import Decoder
# If it fails (Jupyter Notebook), adjust imports dynamically
except ModuleNotFoundError:
    from src.model.encoder import Encoder
    from src.model.decoder import Decoder

import numpy as np
import matplotlib.pyplot as plt

def plot_ecg(ecg_signal, lead_name="Lead"):
    """
    Plots the average ECG signal across the batch.
    
    Parameters:
    ecg_signal (array-like): The ECG waveform batch.
    lead_name (str): The name of the ECG lead being plotted.
    """
    # Ensure the signal is a NumPy array
    if hasattr(ecg_signal, "numpy"):
        ecg_signal = ecg_signal.numpy()

    # Compute the mean ECG waveform across all samples in the batch
    mean_ecg = np.mean(ecg_signal, axis=0)  # Averaging over batch dimension

    print(f"Plotting Mean ECG with shape: {mean_ecg.shape}")  # Debugging output

    plt.figure(figsize=(10, 4))
    plt.plot(mean_ecg, label=f"Mean ECG - {lead_name}", color='r', linewidth=2)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"Average ECG Signal for {lead_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(r"C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VECG\{}.png".format(lead_name))



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Helper:

    @staticmethod
    def generate_paths(paths):
        for path in paths:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
            except FileExistsError:
                print(f"Directory already exists: {path}")



    def data_generator(dataset, method='continue', lead='I'):
        k = 0
        n = len(dataset)
        iterator = iter(dataset[k]) #his allows fetching batches over the dataset sequentially using next(iterator).

        while True:
            try:
                batch = next(iterator)
                ecg_batch = batch['ecg'].get(lead, None)  # Use .get() to prevent KeyError
                # ✅ Convert NaNs back to real zeros BEFORE processing
                ecg_batch = tf.where(tf.math.is_nan(ecg_batch), tf.zeros_like(ecg_batch), ecg_batch)

                # ✅ Check for NaNs after loading data
                if ecg_batch is None or tf.math.reduce_any(tf.math.is_nan(ecg_batch)):
                    print(f"⚠️ NaN detected IMMEDIATELY AFTER LOADING in Lead {lead}!")
                    print(f"Batch Data: {ecg_batch.numpy()}")
                    continue  # Skip this batch



                # ✅ Ensure dtype is float32
                ecg_batch = tf.cast(ecg_batch, dtype=tf.float32)

                # ✅ Check for NaNs again after casting
                if tf.math.reduce_any(tf.math.is_nan(ecg_batch)):
                    print(f"⚠️ NaN detected AFTER CASTING in Lead {lead}!")
                    print(f"Batch Data: {ecg_batch.numpy()}")
                    continue  # Skip this batch

                yield (ecg_batch,)

            except StopIteration:
                if method == 'continue':
                    k = (k + 1) % n  # Wrap around
                    iterator = iter(dataset[k])
                elif method == 'stop':
                    print("✅ Generator exhausted normally.")
                    break  # <---- Use break instead of return (better in tf context)
            except Exception as e:
                print(f"❌ Unhandled error in generator: {e}")
                continue



    @staticmethod
    def get_sample(dataset, n, lead = 'I', label=None):

        k = None
        for example in dataset.take(1):
            k = example

        return (k['ecg'][lead][n:(n + 1)], k[label][n:(n + 1)]) if label else k['ecg'][lead][n:(n + 1)]

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * tf.math.exp(-0.5)

    @staticmethod
    def load_yaml_file(path):
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def write_json_file(file, filepath):
        with codecs.open(filepath, 'w', 'utf8') as f:
            f.write(json.dumps(file, sort_keys=True, ensure_ascii=False))

    @staticmethod
    def print_available_gpu():
        print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


    @staticmethod
    def get_labels(dataset):
        df = pd.DataFrame()
        for data in dataset:
            keys = set(data.keys()) - {'ecg', 'quality'}
            dict_data = {key: data[key].numpy() if hasattr(data[key], 'numpy') else data[key] for key in keys}

            # Ensure `diagnostic` is flattened to a single row
            if "diagnostic" in dict_data:
                diagnostic_array = np.array(dict_data["diagnostic"])
                if diagnostic_array.ndim > 1:  # Flatten multi-dimensional data
                    diagnostic_array = diagnostic_array.flatten()
                dict_data["diagnostic"] = diagnostic_array

            # Convert dictionary to DataFrame and append
            df = pd.concat([df, pd.DataFrame([dict_data])], ignore_index=True)
            
        return df



    @staticmethod
    def get_embedding(model, dataset, split='train', save_path=None, batch_size=512):
        data_train = tfds.load(dataset, split=[split])
        train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
        labels = Helper.get_labels(train)
        z_mean, z_log_var = model._encoder.predict(Helper.data_generator([train], method='stop'))
        z = model.reparameterize(z_mean, z_log_var)

        z_mean = np.expand_dims(z_mean, axis=2)
        z_log_var = np.expand_dims(z_log_var, axis=2)
        z = np.expand_dims(z, axis=2)
        z = np.concatenate((z_mean, z_log_var, z), axis=2)

        if save_path != None:
            file_path = save_path + '/' + dataset + '_' + str(split) + '_data.npy'
            Helper.generate_paths([save_path])
            labels.to_csv(save_path + '/' + dataset + '_' + str(split) + '_labels.csv', index=False)
            with open(file_path, 'wb') as f:
                np.save(f, z)
        return z, labels

    @staticmethod
    def embedding(df, labels, method=PCA(n_components=2)):
        x = method.fit_transform(df)
        x = pd.DataFrame(x)
        x = pd.concat([x, labels], axis=1)
        return x

    @staticmethod
    def load_embedding(path, dataset, split):
        X = np.load(path + dataset + '/' + split + '/' + dataset + '_' + split + '_data.npy')
        latent_dim = X.shape[1]
        y = pd.read_csv(path + '/' + dataset + '/' + split + '/' + dataset + '_' + split + '_labels.csv')
        df = pd.DataFrame(X[:, :, 0])
        df = pd.concat([df, y], axis=1)
        return df, latent_dim

    @staticmethod
    def load_multiple_datasets(datasets):
        size = 0
        data_list = []

        # Ensure split is a list (for both train and val cases)
        splits = datasets['split']
        if isinstance(splits, str):
            splits = [splits]

        for i, k in enumerate(datasets['name']):
            print(f"📥 Loading dataset: {k} (Splits: {splits})")

            # Load and merge multiple splits if needed
            datasets_combined = []
            for split in splits:
                temp = tfds.load(k, split=split, shuffle_files=True)
                datasets_combined.append(temp)

            # Concatenate all splits into a single dataset
            dataset_raw = datasets_combined[0]
            for ds in datasets_combined[1:]:
                dataset_raw = dataset_raw.concatenate(ds)

            data = dataset_raw.shuffle(datasets['shuffle_size']).batch(datasets['batch_size']).prefetch(tf.data.AUTOTUNE)

            size += len(data)
            data_list.append(data)

        return data_list, size



    @staticmethod
    def load_dataset(dataset):
        temp = tfds.load(dataset['name'],split=[dataset['split']], shuffle_files=True)
        data = temp[0].shuffle(dataset['shuffle_size']).batch(dataset['batch_size']).prefetch(tf.data.AUTOTUNE)
        return data, len(temp[0])

    @staticmethod
    def feature_axis_mapping(embeddings, ld):
        n = len(embeddings)
        struct = []
        for ind in range(0, n):
            df = embeddings[ind].fillna(0.0)
            cols = df.columns
            X = np.array(df.iloc[:, 0:ld]).reshape(-1, ld)
            for j in cols[ld:]:
                y = df.loc[:, j]
                if len(np.unique(np.array(y))) > 1:
                    for k in range(0, ld):
                        reg = LinearRegression().fit(X[:, k].reshape(-1, 1), y)
                        score = reg.score(X[:, k].reshape(-1, 1), y)
                        struct.append({'Dim': k, 'Feature': j, 'Score': score})
        return struct

    @staticmethod
    def readable_axis_mapping(struct):
        merged = {}

        for item in struct:
            dim_key = f'Dim {item["Dim"]}'
            if dim_key not in merged:
                merged[dim_key] = {'Features': [], 'Scores': []}
            index = 0
            for i, score in enumerate(merged[dim_key]['Scores']):
                if item['Score'] > score:
                    index = i
                    break
                elif i == len(merged[dim_key]['Scores']) - 1:
                    index = i + 1
            merged[dim_key]['Features'].insert(index, item['Feature'])
            merged[dim_key]['Scores'].insert(index, item['Score'])
        return merged

    @staticmethod
    def axis_feature_mapping(df, ld):
        cols = df.columns
        for j in cols[ld:]:
            max_score = 0.0
            dimension = None
            for k in cols[:ld]:
                X = np.array(df.loc[:, k]).reshape(-1, 1)
                y = df.loc[:, j]
                try:
                    reg = LinearRegression().fit(X, y)
                    score = reg.score(X, y)
                    if score > max_score:
                        dimension = k
                        max_score = score
                except:
                    continue

            print('%20s' % j, '-', dimension, ':', '\t', np.round(max_score, 10))

    @staticmethod
    def experiments(datasets, path, filter='2024-01-01 00:00:00'):
        df = pd.DataFrame()
        print('Executing experiments function')
        subdirectories = glob.glob(os.path.join(path, "*/"))

        for k in subdirectories:
            params = Helper.load_yaml_file(k + 'params.json')
            try:
                train_progress = pd.read_csv(k + 'training/training_progress.csv')
                n = len(train_progress) - 1
                train_progress = train_progress.loc[n:n,
                                 ['alpha', 'beta', 'gamma', 'loss', 'recon', 'mi', 'tc', 'dw_kl']]
                folder_name = os.path.basename(os.path.normpath(k))  
                train_progress['time'] = pd.to_datetime(folder_name, format='%Y-%m-%d_%H-%M-%S')
                train_progress['latent_dim'] = params['latent_dimension']
                train_progress['epoch'] = n + 1
                df = pd.concat([df, train_progress])
            except Exception as e:
                print('failed on:', k, e)
                continue
        df.reset_index(inplace=True, drop=True)
        df = df[(df.time > filter)].sort_values('time').reset_index(drop=True)
        df['total'] = df.recon + df.mi + df.tc + df.dw_kl

        for i, val in enumerate(df.time):
            # Convert time value to string and clean it
            val_str = str(val).replace(' ', '_').replace(':', '-')

            # Properly join path elements
            model_path = os.path.join(path, val_str, "model_best.keras")

            print(f"Loading model from: {model_path}")  # Debugging print

            model = tf.keras.models.load_model(model_path, custom_objects={"Encoder": Encoder, "Decoder": Decoder})
            print("\n📌 Checking Model Layers:")
            for layer in model.layers:
                print(f"- {layer.name} | Type: {type(layer)}")

            emb, ld = Helper.get_embeddings(model, datasets)


            mus_train = np.array(emb[0].iloc[:, :ld])
            ys_train = np.array(emb[0].loc[:, ['p_height', 't_height']])
            df.loc[i, 'MIG'] = Disentanglement.compute_mig(mus_train, ys_train)['discrete_mig']

        return df

    @staticmethod
    def calculate_distances(q, ld):
        mean = np.mean(q.iloc[:, 0:ld], axis=0)
        return np.mean(np.sqrt(np.sum(np.power(q.iloc[:, 0:ld] - mean, 2), axis=1)))

    @staticmethod
    def select_path(path):
        return widgets.Dropdown(
            options=sorted(glob.glob(path + '*/')),
            description='Base path:',
            disabled=False,
        )

    @staticmethod
    def number_to_category(df):
        df.diagnosis = df.diagnosis.replace(
            0.0, 'avblock'
        ).replace(
            1.0, 'fam'
        ).replace(
            2.0, 'iab'
        ).replace(
            3.0, 'lae'
        ).replace(
            4.0, 'lbbb'
        ).replace(
            5.0, 'mi'
        ).replace(
            6.0, 'rbbb'
        ).replace(
            7.0, 'sinus'
        )
        return df

    @staticmethod
    def reparameterize(mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return tf.add(mean, tf.multiply(eps, tf.exp(log_var * 0.5)), name="sampled_latent_variable")

    @staticmethod
    def get_embeddings(models, datasets):
        """
        Generates merged embeddings from multiple models (one per lead),
        handling different latent dimensions and multiple TFDS splits.
        """
        split = datasets['split']
        batch_size = datasets['batch_size']
        results = []

        for dataset in datasets['name']:
            print(f"\n📦 Loading dataset: {dataset}")

            # --- Load and merge multiple splits if provided ---
            if isinstance(split, list):
                print(f"  ⤷ Using splits: {split}")
                combined = tfds.load(dataset, split=split[0], shuffle_files=False)
                for s in split[1:]:
                    combined = combined.concatenate(tfds.load(dataset, split=s, shuffle_files=False))
            else:
                combined = tfds.load(dataset, split=split, shuffle_files=False)

            train_for_labels = combined.batch(batch_size).unbatch().prefetch(tf.data.AUTOTUNE)
            labels = Helper.get_labels(train_for_labels)

            train = combined.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            embeddings_per_sample = None
            total_latent_dim = 0

            for lead, model in zip(lead_names, models):
                print(f"🔍 Processing lead {lead}...")

                df = model._encoder.predict(Helper.data_generator([train], lead=lead, method='stop'))
                df = df[0]  # Extract embeddings
                ld = df.shape[1]
                total_latent_dim += ld

                if embeddings_per_sample is None:
                    embeddings_per_sample = df
                else:
                    embeddings_per_sample = np.hstack((embeddings_per_sample, df))

            print(f"✅ All leads processed. Final embedding shape: {embeddings_per_sample.shape}")
            print(f"📐 Total latent dimension: {total_latent_dim}")

            # Attach labels
            labels.index = range(0, len(labels))
            df_final = pd.concat([pd.DataFrame(embeddings_per_sample), labels], axis=1)
            results.append(df_final)

        return results, total_latent_dim
    
    @staticmethod
    def get_embeddings_single_model(model, datasets, lead):
        """
        Generate embeddings using a single model for a specific lead.
        """

        split = datasets['split']
        batch_size = datasets['batch_size']
        result = []

        for dataset in datasets['name']:
            if isinstance(split, list):
                print(f"\n📦 Loading dataset '{dataset}' with splits: {split}")
                combined = tfds.load(dataset, split=split[0], shuffle_files=False)
                for s in split[1:]:
                    combined = combined.concatenate(tfds.load(dataset, split=s, shuffle_files=False))
            else:
                print(f"\n📦 Loading dataset '{dataset}' with split: {split}")
                combined = tfds.load(dataset, split=split, shuffle_files=False)

            train_for_labels = combined.batch(batch_size).unbatch().prefetch(tf.data.AUTOTUNE)
            labels = Helper.get_labels(train_for_labels)

            train = combined.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # Get embeddings
            print(f"🔍 Processing lead: {lead}")
            df = model._encoder.predict(Helper.data_generator([train], lead=lead, method='stop'))
            df = df[0]  # Extract embeddings array
            ld = df.shape[1]

            # Combine embeddings with labels
            labels.index = range(0, len(labels))
            df_final = pd.concat([pd.DataFrame(df), labels], axis=1)
            result.append(df_final)

        return result, ld

    @staticmethod
    def get_embeddings_multiple_model(models, datasets, lead, save_dir=None):
        """
        Generate and optionally save embeddings using multiple models.
        Optimized for batching and GPU acceleration.
        """
        import time
        import tensorflow_datasets as tfds

        split = datasets['split']
        batch_size = datasets['batch_size']
        result = []
        latent_dim = None

        # Load & batch dataset ONCE
        print(f"\n📦 Loading dataset(s): {datasets['name']} | Splits: {split}")
        all_data = []
        for dataset_name in datasets['name']:
            ds = tfds.load(dataset_name, split=split[0], shuffle_files=False)
            for s in split[1:]:
                ds = ds.concatenate(tfds.load(dataset_name, split=s, shuffle_files=False))
            all_data.append(ds)

        # Merge if needed
        dataset = all_data[0]
        for ds in all_data[1:]:
            dataset = dataset.concatenate(ds)

        # Batch dataset for model prediction
        batched_dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Extract labels once (could be optimized based on your label structure)
        label_dataset = dataset.map(lambda x: x['diagnostic'])  # adjust key as needed
        labels = list(label_dataset.as_numpy_iterator())
        labels = pd.DataFrame(labels)

        # Run embedding generation for each model
        for idx, model in enumerate(models):
            print(f"\n🧠 Running model {idx+1}/{len(models)} on GPU (if available)...")
            start = time.time()

            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                embeddings = model._encoder.predict(
                    Helper.data_generator([batched_dataset], lead=lead, method='stop'),
                    batch_size=batch_size,
                    verbose=1
                )[0]  # Embeddings array

            latent_dim = embeddings.shape[1]
            df_embed = pd.DataFrame(embeddings)
            df_embed.columns = [f"z{i}" for i in range(latent_dim)]
            df_final = pd.concat([df_embed, labels.reset_index(drop=True)], axis=1)
            result.append(df_final)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                df_final.to_pickle(os.path.join(save_dir, f"embeddings_model_{idx}.pkl"))
                print(f"💾 Saved embeddings for model {idx} to disk.")

            print(f"✅ Model {idx} done in {time.time() - start:.2f} seconds")

        return result, latent_dim
    @staticmethod
    def get_icentia_embedding(splits, model):
        datasets = {
            'name': ['icentia11k'],
            'shuffle_size': 1024,
            'batch_size': 1024,
        }
        df = pd.DataFrame()
        for i, k in enumerate(splits):
            datasets.update({'split': k})
            df_pers = Helper.get_embeddings(model, datasets)
            df_pers = df_pers[0][0]
            df_pers['subject'] = k
            df = pd.concat([df, df_pers])
        df.beat = df.beat.replace(0.0, 'Normal').replace(1.0, 'Unclassified').replace(2.0, 'PAC').replace(3.0, 'PVC')
        df = df[df.beat != 'Unclassified']
        return df


    @staticmethod
    def cross_validation_knn(X_train, X_val, y_train_labels, y_val_labels, scoring='accuracy'):
        """
        Performs cross-validation to find the best k for KNN, with additional debugging checks.

        Args:
            X_train (np.array): Training feature matrix (num_samples, 96).
            X_val (np.array): Validation feature matrix.
            y_train_labels (np.array): Training labels (numeric).
            y_val_labels (np.array): Validation labels (numeric).
            scoring (str): Scoring metric (default: 'accuracy').

        Returns:
            best_k (int): The optimal k for KNN.
        """
        from sklearn.model_selection import PredefinedSplit, GridSearchCV
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier

        # ✅ DEBUG: Print initial label info
        print("🔍 Debug: Checking y_train_labels and y_val_labels")
        print("  - Type:", type(y_train_labels))
        print("  - y_train_labels shape:", y_train_labels.shape)
        print("  - y_val_labels shape:", y_val_labels.shape)

        # Combine datasets
        y_combined = np.concatenate((y_train_labels, y_val_labels))
        X_combined = np.vstack((X_train, X_val))
        # Ensure y_combined is a list before saving
        y_combined_list = [str(item) for item in y_combined]

        # Save to a text file for inspection
        file_path = r"C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VECG\analysis\y_combined_output.txt"

        with open(file_path, "w") as f:
            for item in y_combined_list:
                f.write(item + "\n")

        # ✅ DEBUG: Check encoded label distribution
        print("✅ Encoded Labels Shape:", y_combined.shape)
        print("✅ Unique Labels Count:", len(np.unique(y_combined)))
        print("✅ Sample Encoded Labels:", y_combined[:10])

        # Ensure X_combined is float32
        X_combined = X_combined.astype(np.float32)

        # Define PredefinedSplit for validation set
        test_fold = np.concatenate([
            -np.ones(X_train.shape[0]),  # Training samples (-1)
            np.zeros(X_val.shape[0])     # Validation samples (0)
        ])
        ps = PredefinedSplit(test_fold)

        # Imputation (if needed, fills missing values with 0)
        imputer = SimpleImputer(strategy='constant', fill_value=0)

        # KNN Classifier
        knn = KNeighborsClassifier()

        # Pipeline
        pipeline = Pipeline(steps=[('imputer', imputer), ('knn', knn)])

        # Hyperparameter grid search for best k
        param_grid = {'knn__n_neighbors': range(3, 10)}  # Start with a smaller range for debugging

        # ✅ DEBUG: Use error_score='raise' to get exact failure reason
        grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring=scoring, error_score='raise')

        try:
            grid_search.fit(X_combined, y_combined)
            best_k = grid_search.best_params_['knn__n_neighbors']
            best_score = grid_search.best_score_
            print(f'✅ Best k: {best_k} with accuracy: {best_score:.4f}')
        except Exception as e:
            print("🚨 ERROR in GridSearchCV:", str(e))
            return None

        return best_k


    @staticmethod
    def calculate_f1(confusion_matrix, labels):
        num_classes = confusion_matrix.shape[0]
        precision = []
        recall = []
        f1_scores = []
    
        for i in range(num_classes):
            true_positive = confusion_matrix[i, i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive
    
            # Precision: TP / (TP + FP)
            if true_positive + false_positive > 0:
                precision_value = true_positive / (true_positive + false_positive)
            else:
                precision_value = 1
            precision.append(precision_value)
    
            # Recall: TP / (TP + FN)
            if true_positive + false_negative > 0:
                recall_value = true_positive / (true_positive + false_negative)
            else:
                recall_value = 1
            recall.append(recall_value)
    
            # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            if precision_value + recall_value > 0:
                f1_value = 2 * (precision_value * recall_value) / (precision_value + recall_value)
            else:
                f1_value = 1
            f1_scores.append(f1_value)
    
        # Macro F1 score
        macro_f1 = np.mean(f1_scores)
        precision_key = dict(list(zip(labels, precision)))
        recall_key = dict(list(zip(labels, recall)))
        f1_scores_key = dict(list(zip(labels, f1_scores)))
        
        return {
            "Macro F1": macro_f1,
            "Precision": precision_key,
            "Recall": recall_key,
            "F1 Scores": f1_scores_key,
        }

    @staticmethod
    def average_metrics(dict_list, avg_metrics):    
        # Count how many dictionaries we have
        num_dicts = len(dict_list)
        
        # Sum up all the metrics across dictionaries
        for d in dict_list:
            avg_metrics['Macro F1'] += d['Macro F1']
            
            for key in avg_metrics['Precision']:
                avg_metrics['Precision'][key] += d['Precision'].get(key, 0)
                avg_metrics['Recall'][key] += d['Recall'].get(key, 0)
                avg_metrics['F1 Scores'][key] += d['F1 Scores'].get(key, 0)
        
        # Divide by the number of dictionaries to get the averages
        avg_metrics['Macro F1'] /= num_dicts
        
        for key in avg_metrics['Precision']:
            avg_metrics['Precision'][key] /= num_dicts
            avg_metrics['Recall'][key] /= num_dicts
            avg_metrics['F1 Scores'][key] /= num_dicts
        
        return avg_metrics
