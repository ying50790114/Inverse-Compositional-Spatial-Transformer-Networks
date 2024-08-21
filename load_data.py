import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_mnist(batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

    # train val split
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=456)

    train_datasets = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(batchSize).shuffle(x_tr.shape[0])
    val_datasets = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchSize).shuffle(x_val.shape[0])
    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize).shuffle(x_test.shape[0])
    return train_datasets, val_datasets, test_datasets
