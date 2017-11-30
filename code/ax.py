import os
import sys

import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import tensorflow as tf

try:
    import cleverhans
except ImportError:
    print('ERROR: Cannot import cleverhans. Go to '
          'https://github.com/tensorflow/cleverhans to install')
    sys.exit(1)


MNIST_CLASSIFIER = 'mnist_classifier.hd5'


def main():
    dataset = _mnist_dataset()
    classifier = _get_or_retrain_mnist_classifier()
    print(f"Accuracy of {_get_acc(classifier, dataset['test_x']) * 100 :.2f}")


def _get_acc(model, inputs):
    return model.evaluate(inputs, dataset['test_y'])[1]


def _mnist_dataset():
    """Returns a dictionary of the MNIST dataset

    Dictionary contains the inputs and outputs of the training and test set.
    All values are normalized to be between [0., 1.]
    """
    num_classes = 10
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Get all our data in the right shape
    x_train = (x_train.reshape(x_train.shape[0], *input_shape)).astype('float32') / 255.
    x_test = (x_test.reshape(x_test.shape[0], *input_shape)).astype('float32') / 255.
    y_train = (keras.utils.to_categorical(y_train, num_classes)).astype('float32') / 255.
    y_test = (keras.utils.to_categorical(y_test, num_classes)).astype('float32') / 255.
    return {
        'train_x': x_train,
        'train_y': y_train,
        'test_x': x_test,
        'test_y': y_test
    }


def _get_or_retrain_mnist_classifier():
    """Returns a MNIST classifier model

    If the retrain flag is set or the the weights file does not exist, this
    function will retrain the model from scratch. Otherwise the weights will be
    loaded from the file and returned
    """
    if FLAGS.retrain or not os.path.exists(MNIST_CLASSIFIER):
        # Retrain the model from scratch
        dataset = _mnist_dataset()
        model = keras.models.Sequential()
        model.add(Conv2D(32, (5, 5), padding='same',
                  input_shape=dataset['train_x'].shape[1:]))
        model.add(MaxPooling2D(padding='same'))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(MaxPooling2D(padding='same'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(optimizer=keras.optimizers.Adadelta(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(dataset['train_x'], dataset['train_y'], epochs=7, batch_size=50)
        model.save(MNIST_CLASSIFIER)
        return model
    else:
        return keras.models.load_model(MNIST_CLASSIFIER)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_boolean('retrain', False, 'retain models')
    flags.DEFINE_float('eps', 0.15, 'level of perturbation in FGSM')
    FLAGS = flags.FLAGS
    dataset = _mnist_dataset()
    main()

