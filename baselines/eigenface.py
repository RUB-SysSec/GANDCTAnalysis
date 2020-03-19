

import time
from itertools import product
from multiprocessing import pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from classifier import Classifier, read_dataset, subset_dataset
from utils import PersistentDefaultDict

RESULTS_DIR = Path("~/baselines/results")
CLASSIFIER_DIR = RESULTS_DIR.joinpath("classifier")
DATASETS_DIR = Path("~/datasets")

class PCAClassifier(Classifier):

    def __init__(self, pca_target_variance, svm_params, **kwargs):
        super().__init__(**kwargs)
        self.pca = PCA(n_components=pca_target_variance, svd_solver='full')
        self.svm = LinearSVC(**svm_params, max_iter=10000)

    def fit_pca(self, train_data):
        print(f"    -> pca")
        start = time.time()
        self.pca.fit(train_data)
        end = time.time()
        runtime = int(end - start)
        print(f"       pca_components = {self.pca.components_.shape[0]}")
        print(f'       completed in {runtime // 3600}h {(runtime % 3600) // 60}m {(runtime % 60)}s')

    def _fit(self, train_data, train_labels):
        if not hasattr(self.pca, 'components_'):
            self.fit_pca(train_data)
        train_data_pca_transformed = self.pca.transform(train_data)
        self.svm.fit(train_data_pca_transformed, train_labels)

    def _score(self, test_data, test_labels):
        test_data_pca_transformed = self.pca.transform(test_data)
        score = self.svm.score(test_data_pca_transformed, test_labels)
        return score

def generate_params(svm_grid):
    for grid in svm_grid:
        for param_values in product(*tuple(grid.values())):
            params = {}
            for param_name, param_value in zip(grid.keys(), param_values):
                params[param_name] = param_value
            yield params

def grid_search(train_samples, val_samples):
    
    # datasets
    dataset_names = [ "lsun_raw_color", 
                      "lsun_color_log_scaled_normalized", 
                      "celebA_raw_color", 
                      "celebA_color_log_scaled_normalized" ]

    # hyperparameter grid
    pca_target_variances = [0.5, 0.75, 0.95]
    svm_grid = [
        { 'C': [0.001, 0.01, 0.1, 1] }
    ]
    
    results = PersistentDefaultDict(RESULTS_DIR.joinpath(f'eigenfaces_grid_search_train.{train_samples}_val.{val_samples}.json'))

    for dataset_name, pca_target_variance in product(dataset_names, pca_target_variances):
        print(f"\n{dataset_name.upper()} @ {pca_target_variance}")

        # load data
        train_data, train_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', train_samples)
        train_data_pca, _ = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', 10000)
        val_data, val_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_val', val_samples)

        # enumerate svm params
        for svm_params in generate_params(svm_grid):
            svm_params_str = ("_".join([ f'{k}.{v}'for k,v in svm_params.items()])).lower()
            params_str = f'pca_target_variance.{pca_target_variance}_{svm_params_str}'
            print(f"[+] {params_str}")

            # skip if result already exists
            if dataset_name in results.as_dict() and \
               params_str in results.as_dict()[dataset_name]:
                continue

            # train and test classifier
            pca = PCAClassifier(pca_target_variance, svm_params)
            pca.fit_pca(train_data_pca)
            pca.fit(train_data, train_labels)  
            score = pca.score(val_data, val_labels)

            # store result
            results[dataset_name, params_str] = score
    
    print(f"\n[+] Best Results")
    for dataset_name in dataset_names:
        # print best results
        params, acc = sorted(results.as_dict()[dataset_name].items(), key=lambda e: e[1]).pop()
        print(f'    -> {dataset_name}')
        print(f'       {params} @ {acc}')


def train_classifiers(train_samples):

    dataset_names = [ "lsun_raw_color", 
                      "lsun_color_log_scaled_normalized", 
                      "celebA_raw_color", 
                      "celebA_color_log_scaled_normalized" ]

    for dataset_name in dataset_names:
        classifier_name = f'eigenfaces_{dataset_name}.{train_samples}'
        print(f"\n{classifier_name.upper()}")
        # load data
        train_data, train_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', train_samples)
        train_data_pca, _ = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', 10000)
        val_data, val_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_test', 10000)
        # train
        pca = PCAClassifier(pca_target_variance=0.95, svm_params={'C': 1})
        pca.fit_pca(train_data_pca)
        pca.fit(train_data, train_labels)  
        pca.save(CLASSIFIER_DIR.joinpath(f'{classifier_name}.pickle'))
        # test
        pca =  PCAClassifier.load(CLASSIFIER_DIR.joinpath(f'{classifier_name}.pickle'))
        score = pca.score(val_data, val_labels)
        print(f"-> Score {score}")

def test_classifiers():

    dataset_names = [ "lsun_raw_color", 
                      "lsun_color_log_scaled_normalized", 
                      "celebA_raw_color", 
                      "celebA_color_log_scaled_normalized" ]

    results = PersistentDefaultDict(RESULTS_DIR.joinpath(f'final_eigenfaces.json'))

    for dataset_name in dataset_names:
        classifier_name = f"eigenfaces_{dataset_name}.100000"
        print(f"\n{classifier_name.upper()}")
        # load data
        test_data, test_labels = read_dataset(DATASETS_DIR.joinpath(f'{dataset_name}_test'))
        # load classifier
        pca = PCAClassifier.load(CLASSIFIER_DIR.joinpath(f'{classifier_name}.pickle'))
        # score
        score = pca.score(test_data, test_labels)
        results[classifier_name] = score


if __name__ == "__main__":
    
    # print("\n### Eigenfaces Grid Search")
    # grid_search(train_samples=100_000, val_samples=10_000)

    # print("\n### Eigenfaces Train Classifiers")
    # train_classifiers(train_samples=100_000)

    print("\n### Eigenfaces Test Classifiers")
    test_classifiers()
