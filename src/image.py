import tensorflow as tf


def dct2(array, batched=True):
    """DCT-2D transform of an array, by first applying dct-2 along rows then columns

    Arguments:
        array - the array to transform.

    Returns:
        DCT2D transformed array.
    """
    shape = array.shape
    dtype = array.dtype
    array = tf.cast(array, tf.float32)

    if batched:
        # tensorflow computes over last axis (-1)
        # layout (B)atch, (R)ows, (C)olumns, (V)alue
        # BRCV
        array = tf.transpose(array, perm=[0, 3, 2, 1])
        # BVCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 1, 3, 2])
        # BVRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 3, 1])
        # BRCV
    else:
        # RCV
        array = tf.transpose(array, perm=[2, 1, 0])
        # VCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 1])
        # VRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[1, 2, 0])
        # RCV

    array = tf.cast(array, dtype)

    array.shape.assert_is_compatible_with(shape)

    return array


def load_image(path, grayscale=False):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x)
    x = tf.cast(x, tf.float32)
    if grayscale:
        x = tf.image.rgb_to_grayscale(x)
    return x
