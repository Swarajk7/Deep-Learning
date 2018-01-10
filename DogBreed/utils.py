from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def read_all_training_file(train_folder, label_file, image_size=90):
    train_info = pd.read_csv(label_file)
    labels_series = pd.Series(train_info['breed'])
    labels_one_hot = np.asarray(pd.get_dummies(labels_series, sparse=True))
    # X,Y
    X_train = []
    Y_train = []

    file_path_template = train_folder.strip('/') + '/{}.jpg'

    ind = 0
    for f, breed in tqdm(train_info.values):
        f_path = file_path_template.format(f)
        img = cv2.resize(cv2.imread(f_path), (image_size, image_size))
        X_train.append(img)
        Y_train.append(labels_one_hot[ind])
        ind += 1
        #if ind>=100: break
    X_train_norm = np.array(X_train, np.float32) / 255
    Y_train = np.array(Y_train, np.float32)

    return (X_train_norm, Y_train)


def split_train_valid(data, test_percentage=0.1):
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        data[0], data[1], test_size=test_percentage, random_state=1)
    return (X_train, X_valid, Y_train, Y_valid)
