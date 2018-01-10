import vgg19
import utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


X_train, X_valid, Y_train, Y_valid = utils.split_train_valid(
    utils.read_all_training_file(train_folder='./input/train/', label_file='./input/labels.csv'))
num_of_clases = 120  # Y_train.shape[1]
model = vgg19.build_vgg19_model(num_of_clases)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x=X_train, y=Y_train, batch_size=64,
          epochs=5, validation_data=(X_valid, Y_valid))
