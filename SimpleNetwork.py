import tflearn
# from image_utils import data_augmentation, data_preprocessing
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


def data_preprocessing():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    return img_prep


def data_augmentation():
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    return img_aug


def nn():

    # building Deep Learning Network
    input_layer = tflearn.input_data(shape=[None, 128, 128, 3])
    network = tflearn.fully_connected(input_layer, 64, activation='tanh')
    network = tflearn.fully_connected(network, 2, activation='softmax')

    # regression using SGD with learning rate decay
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    network = tflearn.regression(network, optimizer=sgd, metric='accuracy',
                                 loss='categorical_crossentropy')
    return network
