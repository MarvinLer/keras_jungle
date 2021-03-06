from __future__ import print_function, division
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation, add, GlobalAveragePooling2D
# usage_example() function imports
from keras.datasets import cifar10
from keras.utils import to_categorical


def resnet18(input_shape, number_classes):
    def building_block(inputs, n_filters, downsample=False):
        """
        The building block architecture:
        - 3x3 conv 1 stride n filters with relu
        - 3x3 conv 1 stride n filters no activation
        - Add tensor inputs
        - relu
        The conv layers are of stride 1x1 except when the block is downsampling the size of its inputs (counterwisely
        increasing the size of filters).

        :param inputs: tensor containing the input image (or minibatch of images)
        :param n_filters: the two conv layers number of filters
        :param downsample: True to downsample the input (number filters *= 2, output image size /= 2), else False
        :return: output block tensor
        """
        x = Conv2D(n_filters, (3, 3), strides=(1, 1) if not downsample else (2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same')(x)
        if not downsample:
            x = add([inputs, x])
        x = Activation('relu')(x)
        return x

    input_image = Input(shape=input_shape)
    # First block
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_image)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Second block
    x = building_block(x, n_filters=64)
    x = building_block(x, n_filters=64)

    # Third block
    x = building_block(x, n_filters=128, downsample=True)
    x = building_block(x, n_filters=128)

    # Fourth block
    x = building_block(x, n_filters=256, downsample=True)
    x = building_block(x, n_filters=256)

    # Fifth block
    x = building_block(x, n_filters=512, downsample=True)
    x = building_block(x, n_filters=512)

    # Average pooling then one fully connected
    x = GlobalAveragePooling2D()(x)
    x = Dense(number_classes, activation='softmax')(x)

    return Model(inputs=input_image, outputs=x)


def usage_example():
    image_shape = (32, 32, 3)
    num_classes = 10
    batch_size = 16
    epochs = 10

    # loads cifar10 + resize pixels into [0, 1] + one-hot labels
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = resnet18(input_shape=image_shape, number_classes=num_classes)
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


if __name__ == '__main__':
    usage_example()
