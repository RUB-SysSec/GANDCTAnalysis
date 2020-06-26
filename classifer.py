import argparse
import datetime as dt
import os

import numpy as np
import tensorflow as tf

from src.dataset import deserialize_data
from src.models import (build_multinomial_regression,
                        build_multinomial_regression_l1,
                        build_multinomial_regression_l1_l2,
                        build_multinomial_regression_l2, build_resnet,
                        build_simple_cnn, build_simple_nn)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Upsampling / FFHQ
# TRAIN_SIZE = 20_000
# VAL_SIZE = 2_000
# TEST_SIZE = 10_000

# complete size
TRAIN_SIZE = 500_000
VAL_SIZE = 50_000
TEST_SIZE = 150_000

CLASSES = 5
CHANNEL_DIM = 3
INPUT_SHAPE = [128, 128, CHANNEL_DIM]

# Fix for consistent results
tf.random.set_seed(1)


def load_tfrecord(path, train=True, unbounded=True):
    """Load tfrecords."""
    raw_image_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_image_dataset.map(lambda x: deserialize_data(
        x, shape=INPUT_SHAPE), num_parallel_calls=AUTOTUNE)
    if train:
        dataset = dataset.take(TRAIN_SIZE)

    dataset = dataset.batch(BATCH_SIZE)

    if unbounded:
        dataset = dataset.repeat()
    return dataset.prefetch(AUTOTUNE)


def build_model(args):
    input_shape = INPUT_SHAPE
    mirrored_strategy = tf.distribute.MirroredStrategy()
    learning_rate = 0.001

    # select model
    with mirrored_strategy.scope():
        if args.MODEL == "resnet":
            model = build_resnet(input_shape, CLASSES)
        elif args.MODEL == "nn":
            model = build_simple_nn(input_shape, CLASSES, l2=args.l2)
        elif args.MODEL == "cnn":
            model = build_simple_cnn(input_shape, CLASSES)
        elif args.MODEL == "log":
            model = build_multinomial_regression(
                input_shape, CLASSES)
        elif args.MODEL == "log1":
            model = build_multinomial_regression_l1(
                input_shape, CLASSES, l_1=args.l1)
        elif args.MODEL == "log2":
            model = build_multinomial_regression_l2(
                input_shape, CLASSES, l_2=args.l2)
        elif args.MODEL == "log3":
            model = build_multinomial_regression_l1_l2(
                input_shape, CLASSES, l_1=args.l1, l_2=args.l2)
        else:
            raise NotImplementedError(
                "Error model you selected not available!")

        if CLASSES == 1:
            loss = tf.keras.losses.binary_crossentropy
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy
        metrics = ["acc"]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, ),
                      loss=loss,
                      metrics=metrics)

    model_name = f"{args.MODEL}_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_batch_{args.batch_size}_learning_rate_{learning_rate}"
    return model, model_name


def train(args):
    train_dataset = load_tfrecord(args.TRAIN_DATASET)
    val_dataset = load_tfrecord(args.VAL_DATASET)

    model, model_name = build_model(args)

    log_path = f"./log/{model_name}"
    ckpt_dir = f"./ckpt/{model_name}/"
    model_dir = f"./final_models/{model_name}/"
    os.makedirs(ckpt_dir)
    os.makedirs(model_dir)

    update_freq = 50

    if args.debug:
        callbacks = None
    else:
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_path,
                update_freq=update_freq,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=args.early_stopping,
                restore_best_weights=True,
            ),
        ]

    model.summary()
    model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
              validation_data=val_dataset,
              validation_steps=VAL_SIZE // BATCH_SIZE,
              callbacks=callbacks)

    _, eval_accuracy = model.evaluate(
        val_dataset, steps=VAL_SIZE // BATCH_SIZE, verbose=0)

    return model, eval_accuracy, model_dir


def train_and_save_model(args):
    model, eval_accuracy, model_dir = train(args)

    print(
        f"Saving model with accuracy - {eval_accuracy:.2%} - to {model_dir}")
    model.save(model_dir, save_format="tf")


def test(args):
    test_dataset = load_tfrecord(args.TEST_DATASET, train=False)

    # load model
    model = tf.keras.models.load_model(args.MODEL)
    model.summary()
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)


def main(args):
    args.grayscale = True
    if args.mode == "train":
        train_and_save_model(args)
    elif args.mode == "test":
        test(args)
    else:
        raise NotImplementedError("Specified non valid mode!")


def parse_args():
    global BATCH_SIZE, INPUT_SHAPE, CLASSES, CHANNEL_DIM
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", "-s", help="Images to load.", type=int, default=None)

    commands = parser.add_subparsers(help="Mode {train|test}.", dest="mode")

    train = commands.add_parser("train")
    epochs = 50
    train.add_argument(
        "MODEL", help="Select model to train {resnet, cnn, nn, log, log1, log2, log3}.", type=str)
    train.add_argument("TRAIN_DATASET", help="Dataset to load.", type=str)
    train.add_argument("VAL_DATASET", help="Dataset to load.", type=str)
    train.add_argument("--debug", "-d", help="Debug mode.",
                       action="store_true")
    train.add_argument(
        "--epochs", "-e", help=f"Epochs to train for; Default: {epochs}.", type=int, default=epochs)
    train.add_argument("--image_size",
                       help=f"Image size. Default: {INPUT_SHAPE}", type=int, default=128)
    train.add_argument("--early_stopping",
                       help=f"Early stopping criteria. Default: 5", type=int, default=5)
    train.add_argument("--classes",
                       help=f"Classes. Default: {CLASSES}", type=int, default=CLASSES)
    train.add_argument("--grayscale", "-g",
                       help=f"Train on grayscaled images.", action="store_true")
    train.add_argument("--batch_size", "-b",
                       help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=BATCH_SIZE)
    train.add_argument("--l1",
                       help=f"L1 reguralizer intensity. Default: 0.01", type=float, default=0.01)
    train.add_argument("--l2",
                       help=f"L2 reguralizer intensity. Default: 0.01", type=float, default=0.01)

    test = commands.add_parser("test")
    test.add_argument("MODEL", help="Path to model.", type=str)
    test.add_argument("TEST_DATASET", help="Dataset to load.", type=str)
    test.add_argument("--image_size",
                      help=f"Image size. Default: {INPUT_SHAPE}", type=int, default=128)
    test.add_argument("--grayscale", "-g",
                      help=f"Test on grayscaled images.", action="store_true")
    test.add_argument("--batch_size", "-b",
                      help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=BATCH_SIZE)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    if args.grayscale:
        CHANNEL_DIM = 1

    INPUT_SHAPE = [args.image_size, args.image_size, CHANNEL_DIM]

    if "classes" in args:
        CLASSES = args.classes

    return args


if __name__ == "__main__":
    main(parse_args())
