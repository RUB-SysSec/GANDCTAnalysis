
import time
from itertools import product
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from classifier import Classifier, read_dataset
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from utils import PersistentDefaultDict


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

    @staticmethod
    def generate_params(svm_grid):
        for grid in svm_grid:
            for param_values in product(*tuple(grid.values())):
                params = {}
                for param_name, param_value in zip(grid.keys(), param_values):
                    params[param_name] = param_value
                yield params


    @staticmethod
    def grid_search(dataset_name, datasets_dir, output_dir, n_jobs):
        # hyperparameter grid
        pca_target_variances = [0.25, 0.5, 0.75, 0.95]
        svm_grid = [
            { 'C': [0.0001, 0.001, 0.01, 0.1] }
        ]
        
        # init results
        results = PersistentDefaultDict(output_dir.joinpath(f'eigenfaces_grid_search.json'))

        for pca_target_variance in pca_target_variances:
            # enumerate svm params
            for svm_params in PCAClassifier.generate_params(svm_grid):
                svm_params_str = "_".join([ f'{k}.{v}'for k,v in svm_params.items()])
                params_str = f'pca_target_variance.{pca_target_variance}_{svm_params_str}'
                print(f"[+] {params_str}")

                # skip if result already exists
                if dataset_name in results.as_dict() and \
                params_str in results.as_dict()[dataset_name]:
                    continue

                # load data
                train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
                train_data_pca, _ = read_dataset(datasets_dir, f'{dataset_name}_train', subset_to_size=10000)
                val_data, val_labels = read_dataset(datasets_dir, f'{dataset_name}_val')

                # train and test classifier
                pca = PCAClassifier(pca_target_variance, svm_params)
                pca.fit_pca(train_data_pca)
                pca.fit(train_data, train_labels)  
                score = pca.score(val_data, val_labels)

                # store result
                results[dataset_name, params_str] = score
        
        return results

    def train_classifier(dataset_name, datasets_dir, output_dir, n_jobs, pca_target_variance, C):
        # classifier name
        classifier_name = f'classifier_{dataset_name}_eigenfaces_v.{pca_target_variance}_c.{C}'
        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
        train_data_pca, _ = read_dataset(datasets_dir, f'{dataset_name}_train', subset_to_size=10000)
        # train
        pca = PCAClassifier(pca_target_variance=pca_target_variance, svm_params={'C': C})
        pca.fit_pca(train_data_pca)
        pca.fit(train_data, train_labels)  
        pca.save(output_dir.joinpath(f'{classifier_name}.pickle'))
        # test
        PCAClassifier.test_classifier(classifier_name, dataset_name, datasets_dir, output_dir, n_jobs)

    def test_classifier(classifier_name, dataset_name, datasets_dir, output_dir, n_jobs):
        results = PersistentDefaultDict(output_dir.joinpath(f'eigenfaces_test.json'))
        # load data
        test_data, test_labels = read_dataset(datasets_dir, f'{dataset_name}_test')
        # load classifier
        pca = PCAClassifier.load(output_dir.joinpath(f'{classifier_name}.pickle'))
        # score
        score = pca.score(test_data, test_labels)
        results[classifier_name] = score
