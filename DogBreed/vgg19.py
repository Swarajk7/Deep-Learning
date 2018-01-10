from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.applications.vgg19 import preprocess_input
from keras import regularizers
from keras.models import Model
import numpy as np


def build_vgg19_model(num_classes, hidden_layer=1024, drop_out=0.4, image_size=90):
    base_model = VGG19(weights='imagenet', include_top=False,
                       input_shape=(image_size, image_size, 3))  # download weights for VGG19 trained using imagenet dataset and instantiate the model

    X = base_model.output
    #X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(hidden_layer, activation='relu')(X)
    X = Dropout(rate=0.4)(X)
    X = Dense(num_classes, activation='softmax')(X)
    model = Model(input=base_model.inputs, outputs=X, name='VGG19')
    # set all other layers as non-trainable
    for layer in base_model.layers:
        if layer.name != 'block5_conv4':
            layer.trainable = False
        else:
            layer.kernel_regularizer = regularizers.l2(0.01)
    return model
