import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


class Classifier(object):

    def __init__(self):
        super().__init__()

    def _fit(self, train_data, train_labels):
        raise NotImplementedError()

    def _score(self, test_data, test_labels):
        raise NotImplementedError()

    def fit(self, train_data, train_labels):
        print(f"    fit")
        start = time.time()
        self._fit(train_data, train_labels)
        end = time.time()
        runtime = int(end - start)
        print(f'    completed in {runtime // 3600}h {(runtime % 3600) // 60}m {(runtime % 60)}s')
        return self

    def score(self, test_data, test_labels):
        print(f"    score")
        start = time.time()
        score = self._score(test_data, test_labels)
        end = time.time()
        runtime = int(end - start)
        print(f'    -> {score}')
        print(f'    completed in {runtime // 3600}h {(runtime % 3600) // 60}m {(runtime % 60)}s')
        return score

    def save(self, output_path):
        output_dir = Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        output_path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(in_path):
        instance = pickle.loads(Path(in_path).read_bytes())
        return instance

def read_dataset(datasets_dir, dataset_name, subset_to_size=None, flatten=True):
    print(f"[+] Read from {dataset_name}")
    dataset_dir = datasets_dir / dataset_name

    labels = np.load(dataset_dir.joinpath("labels.npy"))

    if not subset_to_size:
        # read full dataset
        imgs = []
        for idx in tqdm(range(labels.size), bar_format='    {l_bar}{bar:30}{r_bar}'):
            img_path = dataset_dir.joinpath(f'{idx:06}.npy')
            img = np.load(img_path)
            imgs.append(img)
        imgs = np.stack(imgs, 0)
        if flatten:
            imgs = imgs.reshape(labels.size, -1)
        return imgs, labels
    
    else:
        # subset dataset
        size_per_label = subset_to_size // np.unique(labels).size

        subset_data = []
        subset_labels = []

        sizes_per_label = defaultdict(int)
        p_bar = tqdm(total=subset_to_size, bar_format='    {l_bar}{bar:30}{r_bar}')
        for idx, label in enumerate(labels):

            if sizes_per_label[label] < size_per_label:
                img_path = dataset_dir.joinpath(f'{idx:06}.npy')
                subset_data.append(np.load(img_path))
                subset_labels.append(label) 
                p_bar.update(1)
                sizes_per_label[label] += 1

            if len(subset_data) == subset_to_size:
                p_bar.close()
                break
            
        else:
            raise Exception('[!] ran out of images')
            
        subset_data = np.stack(subset_data, 0)
        if flatten:
            subset_data = subset_data.reshape(subset_to_size, -1)
        return subset_data, subset_labels
