from __future__ import print_function, division
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation, add, GlobalAveragePooling2D
# usage_example() function imports
from keras.datasets import cifar10
from keras.utils import to_categorical


def resnet34(input_shape, number_classes):
    def bottleneck_building_block(inputs, n_filters, downsample=False):
        """
        The bottleneck building block architecture:
        - 1x1 conv 1 stride n/4 filters (dimension decreasing layer) with relu
        - 3x3 conv 1 stride n/4 filters (bottleneck layer) with relu
        - 1x1 conv 1 stride n filters (dimension increasing layer) no activation
        - Add tensor inputs
        - relu
        The conv layer is of stride 1x1 except when the block is downsampling the size of its inputs (counterwisely
        increasing the size of filters).

        :param inputs: tensor containing the input image (or minibatch of images)
        :param n_filters: the last layer number of filters (should be the number of channels of input except when
            downsampling); the first two have filter size = n_filters // 4 (exc. downsampling)
        :param downsample: True to downsample the input, else False
        :return: output block tensor
        """
        x = Conv2D(n_filters // 4, (1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
        x = Conv2D(n_filters // 4, (3, 3), strides=(1, 1) if not downsample else (2, 2), padding='same',
                   activation='relu')(x)
        x = Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same')(x)
        if not downsample:
            x = add([inputs, x])
        outputs = Activation('relu')(x)

        return outputs

    input_image = Input(shape=input_shape)
    # First block
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_image)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Second block
    x = bottleneck_building_block(x, n_filters=64)
    x = bottleneck_building_block(x, n_filters=64)
    x = bottleneck_building_block(x, n_filters=64)

    # Third block
    x = bottleneck_building_block(x, n_filters=128, downsample=True)
    x = bottleneck_building_block(x, n_filters=128)
    x = bottleneck_building_block(x, n_filters=128)
    x = bottleneck_building_block(x, n_filters=128)

    # Fourth block
    x = bottleneck_building_block(x, n_filters=256, downsample=True)
    x = bottleneck_building_block(x, n_filters=256)
    x = bottleneck_building_block(x, n_filters=256)
    x = bottleneck_building_block(x, n_filters=256)
    x = bottleneck_building_block(x, n_filters=256)
    x = bottleneck_building_block(x, n_filters=256)

    # Fifth block
    x = bottleneck_building_block(x, n_filters=512, downsample=True)
    x = bottleneck_building_block(x, n_filters=512)
    x = bottleneck_building_block(x, n_filters=512)

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

    model = resnet34(input_shape=image_shape, number_classes=num_classes)
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


if __name__ == '__main__':
    usage_example()
