
from collections import defaultdict
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from classifier import Classifier, read_dataset
from prnu_functions import aligned_cc, extract_multiple_aligned, extract_single
from utils import PersistentDefaultDict

class PRNUClassifier(Classifier):

    def __init__(self, levels, sigma, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        self.sigma = sigma
        self.gan_fingerprints = None

    def _fit(self, train_data, train_labels):
        # sort training data by labels
        unique_labels = np.unique(train_labels)
        train_data_by_label = defaultdict(list)
        for img, label in zip(train_data, train_labels):
            train_data_by_label[label].append(img)
        # extract fingerprints
        self.gan_fingerprints = []
        for label in unique_labels:
            imgs = train_data_by_label[label]
            gan_fingerprint = extract_multiple_aligned(imgs, self.levels, self.sigma, processes=cpu_count())
            self.gan_fingerprints.append(gan_fingerprint)

    def _score(self, test_data, test_labels):
        # extract fingerprints
        img_fingerprints = [] 
        for img, label in tqdm(zip(test_data, test_labels), bar_format='    {l_bar}{bar:30}{r_bar}', 
                                                            total=len(test_labels)):
            img_fingerprint = extract_single(img, self.levels, self.sigma)
            img_fingerprints.append(img_fingerprint)
        # correlate images with GAN fingerprints
        cc = aligned_cc(np.stack(self.gan_fingerprints, 0), np.stack(img_fingerprints, 0))['ncc']
        # calculate score
        predictions = np.argmax(cc, axis=0)
        correct = 0
        incorrect = 0
        for prediction, label in zip(predictions, test_labels):
            if str(prediction) == str(label):
                correct += 1
            else:
                incorrect += 1
        score = correct/(correct+incorrect)
        return score

    @staticmethod
    def grid_search(dataset_name, datasets_dir, output_dir, n_jobs):
        # init results
        results = PersistentDefaultDict(output_dir.joinpath(f'prnu_grid_search.json'))

        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train', flatten=False)
        val_data, val_labels = read_dataset(datasets_dir, f'{dataset_name}_val', flatten=False)
        train_data = train_data.astype(np.dtype('uint8'))
        val_data = val_data.astype(np.dtype('uint8'))

        # hyperparameter grid
        levels_range = range(1, 5, 1)
        sigma_range = np.arange(0.05, 1, 0.05)
        
        for levels, sigma in product(levels_range, sigma_range):
            # classifier name
            prnu_params_str = f'levels.{levels}_sigma.{sigma}'
            print(f"[+] {prnu_params_str}")

            # skip if result already exists
            if dataset_name in results.as_dict() and \
                prnu_params_str in results.as_dict()[dataset_name]:
                continue

            # train and test classifier
            prnu = PRNUClassifier(levels, sigma)
            prnu.fit(train_data, train_labels)
            score = prnu.score(val_data, val_labels)

            # store result
            results[dataset_name, prnu_params_str] = score
        
        return results

    @staticmethod
    def train_classifier(dataset_name, datasets_dir, output_dir, n_jobs, levels, sigma):
        # classifier name
        classifier_name = f'classifier_{dataset_name}_prnu_levels.{levels}_sigma.{sigma}'
        print(f"\n{classifier_name.upper()}")
        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train', flatten=False)
        train_data = train_data.astype(np.dtype('uint8'))
        # train
        prnu = PRNUClassifier(levels, sigma)
        prnu.fit(train_data, train_labels)
        prnu.save(output_dir.joinpath(f'{classifier_name}.pickle'))
        # test
        PRNUClassifier.test_classifier(classifier_name, dataset_name, datasets_dir, output_dir, n_jobs)


    def test_classifier(classifier_name, dataset_name, datasets_dir, output_dir, n_jobs):
        print(f"\n{classifier_name.upper()}")
        results = PersistentDefaultDict(output_dir.joinpath(f'prnu_test.json'))
        # load data
        test_data, test_labels = read_dataset(datasets_dir, f'{dataset_name}_test', flatten=False)
        test_data = test_data.astype(np.dtype('uint8'))
        # load classifier
        prnu = PRNUClassifier.load(output_dir.joinpath(classifier_name + '.pickle'))
        # score
        score = prnu.score(test_data, test_labels)
        results[classifier_name] = score
