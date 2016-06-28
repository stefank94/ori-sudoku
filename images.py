import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_one_image(path):
    img = cv2.imread(path, 0)
    img = inverte(img)
    data = np.asarray(img, dtype="uint8")
    numbers = [None]*81
    counter = 0        
    for i in xrange(0,9):       #secemo tablu na 91 sliku
        for j in xrange(0,9):
            num = data[i*33:i*33+33, j*33:j*33+33]
            num = num[4:32, 4:32]
            numbers[counter] = num
            counter += 1
   # for i in range(81):
    #    plt.subplot(9,9,i+1)
     #   plt.imshow(numbers[i], cmap='gray', interpolation='none') #stampa celu tablu
    
    for i in range(81):
        num = numbers[i]
        num = num.reshape(1, 784)
        num = num.astype("float32")
        num /= 255
        numbers[i] = num
    return numbers
        
def get_images_for_learning():
    tables = [None]*60
    for f in range(1,61):
        s = "0"
        if f < 10:
            s = "00"
        img = cv2.imread("slike/easy/sudoku_easy_" + s + str(f) + ".jpg", 0)
        img = inverte(img)
        data = np.asarray(img, dtype="uint8")
        numbers = [None]*81
        counter = 0        
        for i in xrange(0,9):       #secemo tablu na 91 sliku
            for j in xrange(0,9):
                num = data[i*33:i*33+33, j*33:j*33+33]
                num = num[4:32, 4:32]
                numbers[counter] = num
                counter += 1
        
        for i in range(len(numbers)):
            num = numbers[i]
            num = num.reshape(1, 784)
            num = num.astype("float32")
            num /= 255
            numbers[i] = num
        tables[f-1] = numbers
    return tables
   
def get_classes():
    tables = [None]*60
    for f in range(1,61):
        s = "0"
        if f < 10:
            s = "00"
        file = open("slike/easy/sudoku_easy_" + s + str(f) + ".dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    return tables
   
def inverte(imagem):
    imagem = (255-imagem)
    return imagem

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
