
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

from classifier import Classifier, subset_dataset, read_dataset
from utils import PersistentDefaultDict

N_JOBS = 40
RESULTS_DIR = Path("~/baselines/results")
CLASSIFIER_DIR = RESULTS_DIR.joinpath("classifier")
DATASETS_DIR = Path("~/datasets")


class KNNClassifier(Classifier):

    def __init__(self, n_neighbors, n_jobs, **kwargs):
        super().__init__(**kwargs)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)

    def _fit(self, train_data, train_labels):
        self.knn.fit(train_data, train_labels)

    def _score(self, test_data, test_labels):
        return self.knn.score(test_data, test_labels)

def grid_search(train_samples, val_samples):

    # datasets
    dataset_names = [ "lsun_raw_color", 
                      "lsun_color_log_scaled_normalized", 
                      "celebA_raw_color", 
                      "celebA_color_log_scaled_normalized" ]

    # hyperparameter grid
    knn_grid = [1] + [(2**x)+1 for x in range(1, 11)]

    results = PersistentDefaultDict(RESULTS_DIR.joinpath(f'knn_grid_search_train.{train_samples}_val.{val_samples}.json'))

    for dataset_name in dataset_names:
        print(f"\n{dataset_name.upper()}")

        # load data
        train_data, train_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', train_samples)
        val_data, val_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_val', val_samples)
        
        for n_neighbors in knn_grid:
            knn_params_str = f'n_neighbors.{n_neighbors}'
            print(f"[+] {knn_params_str}")

            # skip if result already exists
            if dataset_name in results.as_dict() and \
               knn_params_str in results.as_dict()[dataset_name]:
                continue
            
            # train and test classifier
            knn = KNNClassifier(n_neighbors, N_JOBS)
            knn.fit(train_data, train_labels)
            score = knn.score(val_data, val_labels)

            # store result
            results[dataset_name, knn_params_str] = score

    print(f"\n[+] Best Results")
    for dataset_name in dataset_names:
        # print best results
        params, acc = sorted(results.as_dict()[dataset_name].items(), key=lambda e: e[1]).pop()
        print(f'    -> {dataset_name}')
        print(f'       {params} @ {acc}')

def test_classifiers():

    n_neighbors_config = {
        "lsun_raw_color" : 1,
        "lsun_color_log_scaled_normalized" : 65,
        "celebA_raw_color" : 33,
        "celebA_color_log_scaled_normalized" : 129
    }

    dataset_names = [ "lsun_raw_color", 
                      "lsun_color_log_scaled_normalized", 
                      "celebA_raw_color", 
                      "celebA_color_log_scaled_normalized" ]

    results = PersistentDefaultDict(RESULTS_DIR.joinpath(f'final_knn.json'))

    for dataset_name in dataset_names:
        classifier_name = f"knn_{dataset_name}.100000"
        print(f"\n{classifier_name.upper()}")
        # load data
        train_data, train_labels = subset_dataset(DATASETS_DIR, f'{dataset_name}_train', 100_000)
        test_data, test_labels = read_dataset(DATASETS_DIR.joinpath(f'{dataset_name}_test'))
        # train
        knn = KNNClassifier(n_neighbors_config[dataset_name], N_JOBS)
        knn.fit(train_data, train_labels)
        # score
        score = knn.score(test_data, test_labels)
        results[classifier_name] = score


if __name__ == "__main__":
    
    # print("\n### kNN grid search")
    # grid_search(train_samples=100_000, val_samples=10_000)

    print("\n### kNN Test Classifiers")
    test_classifiers()