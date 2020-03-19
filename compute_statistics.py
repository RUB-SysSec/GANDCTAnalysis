"""A script for computing different statistics in either grayscale or over each color channel separately.
"""
import argparse
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from src.dataset import image_paths
from src.image_np import dct2, load_image
from src.math import log_scale, welford, welford_multidimensional


def _plot(outpath, name, data, align, **kwargs):
    if align:
        # align values for heatmap plots
        max_value = np.asarray(
            list(map(lambda x: x.max(), data))).max()
        min_value = np.asarray(
            list(map(lambda x: x.min(), data))).min()
        kwargs.update({"vmin": min_value, "vmax": max_value})

    fig, axis = plt.subplots(dpi=300)
    mat = axis.matshow(data, **kwargs)

    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    _ = fig.colorbar(mat, ax=axis)

    fig.tight_layout()
    os.makedirs(outpath, exist_ok=True)
    fig.savefig(f"{outpath}/{name}.pdf")
    plt.close(fig=fig)


def plot_without_labels(outpath, datas, align=True, **kwargs):
    for name, data in datas:
        if len(data.shape) > 2 and data.shape[2] == 3:
            _plot(outpath, f"{name}_red", data[:, :, 0], align, **kwargs)
            _plot(outpath, f"{name}_green", data[:, :, 1], align, **kwargs)
            _plot(outpath, f"{name}_blue", data[:, :, 2], align, **kwargs)
        else:
            _plot(outpath, name, data, align, **kwargs)


class Statistics(object):
    """Convenience object for computing statistics.
    """

    def __init__(self, amount, datasets, color, output):
        self.amount = amount
        self.datasets = datasets
        self.color = color
        self.output = output

        self.ref_mean = None
        self.ref_std = None

        self.means = list()
        self.stds = list()
        self.mean_differences = list()

    def compute_and_plot(self):
        self._compute_statistics()
        self._plot()

    def _compute_statistics(self):
        for encoded in self.datasets:
            dataset, name = encoded.split(",")
            dataset = image_paths(dataset)
            dataset = dataset[:self.amount]

            images = map(lambda d: load_image(
                d, grayscale=not self.color), dataset)
            images_dct = map(dct2, images)

            # simple base statistics
            if self.color:
                mean, variance = welford_multidimensional(images_dct)

            else:
                mean, variance = welford(images_dct)

            std = np.sqrt(variance)

            self.means.append((f"mean_{name}", log_scale(mean)))
            self.stds.append((f"std_{name}", log_scale(std)))

            if self.ref_mean is None:
                self.ref_mean = mean
                self.ref_std = std
                continue

            # Other statistics calculated in reference to ref stats
            # mean difference
            mean_diff = np.abs(log_scale(np.abs(self.ref_mean)) -
                               log_scale(np.abs(mean)))
            self.mean_differences.append((f"mean_differnce_{name}", mean_diff))

    def _plot(self):
        # plotting
        outpath = f"{self.output}/{datetime.utcnow().strftime('%d-%m-%Y-%H-%M-%S')}"
        outpath = f"{outpath}/statisticts/"
        os.makedirs(outpath)

        for cm_name, cm in [("inferno", plt.cm.inferno), ("winter", plt.cm.winter), ("pink", plt.cm.pink), ("coolwarm", plt.cm.coolwarm)]:
            cm_outpath = f"{outpath}/{cm_name}"
            plot_without_labels(cm_outpath, self.means, cmap=cm)
            plot_without_labels(cm_outpath, self.stds, cmap=cm)
            plot_without_labels(cm_outpath, self.mean_differences, cmap=cm)


def main(args):
    amount = args.AMOUNT if args.AMOUNT > 0 else None

    Statistics(amount, args.DATASETS, args.color,
               args.output).compute_and_plot()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "AMOUNT", help="The amount of images to load.", type=int)

    parser.add_argument("DATASETS", help="Path to datasets. The first entry is assumed to be the referrence one.",
                        type=str, nargs="*")

    output_default = "output"
    parser.add_argument(
        "--output", "-o", help="Output directory. Default: {output_default}.", type=str, default=output_default)
    parser.add_argument(
        "--color", "-c", help="Plot for each color channel seperate.", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
