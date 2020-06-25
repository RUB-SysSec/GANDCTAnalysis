
from pathlib import Path

from classifier import Classifier, read_dataset
from sklearn.neighbors import KNeighborsClassifier
from utils import PersistentDefaultDict


class KNNClassifier(Classifier):

    def __init__(self, n_neighbors, n_jobs, **kwargs):
        super().__init__(**kwargs)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)

    def _fit(self, train_data, train_labels):
        self.knn.fit(train_data, train_labels)

    def _score(self, test_data, test_labels):
        return self.knn.score(test_data, test_labels)

    @staticmethod
    def grid_search(dataset_name, datasets_dir, output_dir, n_jobs):

        # hyperparameter grid
        knn_grid = [1] + [(2**x)+1 for x in range(1, 11)]

        # init results
        results = PersistentDefaultDict(output_dir.joinpath(f'knn_grid_search.json'))

        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
        val_data, val_labels = read_dataset(datasets_dir, f'{dataset_name}_val')
        
        for n_neighbors in knn_grid:
            knn_params_str = f'n_neighbors.{n_neighbors}'
            print(f"[+] {knn_params_str}")

            # skip if result already exists
            if dataset_name in results.as_dict() and \
                knn_params_str in results.as_dict()[dataset_name]:
                continue
            
            # train and test classifier
            knn = KNNClassifier(n_neighbors, n_jobs)
            knn.fit(train_data, train_labels)
            score = knn.score(val_data, val_labels)

            # store result
            results[dataset_name, knn_params_str] = score

        return results

    def train_classifier(dataset_name, datasets_dir, output_dir, n_jobs, n_neighbors):
        results = PersistentDefaultDict(output_dir.joinpath(f'knn_test.json'))
        # classifier name
        classifier_name = f'classifier_{dataset_name}_knn_n_neighbors.{n_neighbors}'
        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
        test_data, test_labels = read_dataset(datasets_dir, f'{dataset_name}_test')
        # train classifier
        knn = KNNClassifier(n_neighbors, n_jobs)
        knn.fit(train_data, train_labels)
        # test classifier
        score = knn.score(test_data, test_labels)
        results[classifier_name] = score
