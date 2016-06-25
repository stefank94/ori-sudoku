import numpy as np
import cv2


def load_train_x_set():
    return load_x_set(1,51)

def load_train_y_set():
    return load_y_set(1, 51)

def load_test_x_set():
    return load_x_set(51, 61)

def load_test_y_set():
    return load_y_set(51, 61)

def load_x_set(od, do):
    size = do - od
    list = np.zeros((size * 3, 90000))

    i = 0
    for f in range(od, do):
        s = "0"
        if f < 10:
            s = "00"
        img = cv2.imread("slike/easy/sudoku_easy_" + s + str(f) + ".jpg", 0)
        #img = keras.preprocessing.image.load_image("slike/easy/sudoku_easy_" + s + str(i) + ".jpg", True)
        data = np.asarray(img, dtype="uint8")
        list[i, :] = data.ravel()
        i += 1

        img = cv2.imread("slike/medium/sudoku_medium_" + s + str(f) + ".jpg", 0)
        #img = keras.preprocessing.image.load_image("slike/medium/sudoku_medium_" + s + str(i) + ".jpg", True)
        data = np.asarray(img, dtype="uint8")
        list[i, :] = data.ravel()
        i += 1

        img = cv2.imread("slike/hard/sudoku_hard_" + s + str(f) + ".jpg", 0)
        #img = keras.preprocessing.image.load_image("slike/hard/sudoku_hard_" + s + str(i) + ".jpg", True)
        data = np.asarray(img, dtype="uint8")
        list[i, :] = data.ravel()
        i += 1

    return list

def load_y_set(od, do):
    size = do - od
    list = np.zeros((size * 3, 81))

    i = 0
    for f in range(od, do):

        s = "0"
        if f < 10:
            s = "00"

        file = open("slike/easy/sudoku_easy_" + s + str(f) + ".dat", "r")
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char) / 9
                list[i, j] = x
                j += 1
        i += 1

        file = open("slike/medium/sudoku_medium_" + s + str(f) + ".dat", "r")
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char) / 9
                list[i, j] = x
                j += 1
        i += 1

        file = open("slike/hard/sudoku_hard_" + s + str(f) + ".dat", "r")
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char) / 9
                list[i, j] = x
                j += 1
        i += 1

    return list
