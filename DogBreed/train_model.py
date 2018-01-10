import vgg19
import resnet50
import utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# hparams
img_size = 300
batch = 32
num_epochs = 20
hidden_layer = 512

model_name = 'RESNET50'  # 'VGG19','RESNET50'

X_train, X_valid, Y_train, Y_valid = utils.split_train_valid(
    utils.read_all_training_file(
        train_folder='./input/train/', label_file='./input/labels.csv', image_size=img_size),
    test_percentage=0.2)
num_of_clases = 120 #Y_train.shape[1]

if model_name == 'VGG19':
    model = vgg19.build_model(
        num_of_clases, image_size=img_size, hidden_layer=hidden_layer)
elif model_name == 'RESNET50':
    model = resnet50.build_model(
        num_of_clases, image_size=img_size, hidden_layer=hidden_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
# model.fit(x=X_train, y=Y_train, batch_size=batch,
#          epochs=num_epochs, validation_data=(X_valid, Y_valid))

# data augmentation
generator = utils.augment_image_generator()
train_generator = generator.flow(
    x=X_train, y=Y_train, batch_size=batch, shuffle=True)

#fit model
model.fit_generator(generator=train_generator,
                    steps_per_epoch=X_train.shape[0] // batch,
                    epochs=num_epochs,
                    validation_data=(X_valid, Y_valid))