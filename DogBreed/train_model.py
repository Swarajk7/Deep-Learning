import vgg19
import utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


#hparams
img_size = 150
batch = 16
num_epochs = 5


X_train, X_valid, Y_train, Y_valid = utils.split_train_valid(
    utils.read_all_training_file(
        train_folder='./input/train/', label_file='./input/labels.csv',image_size=img_size),
    test_percentage=0.2)
num_of_clases = Y_train.shape[1]
model = vgg19.build_vgg19_model(num_of_clases,image_size=img_size)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x=X_train, y=Y_train, batch_size=batch,
          epochs=num_epochs, validation_data=(X_valid, Y_valid))
