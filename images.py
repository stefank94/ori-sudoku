import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

def get_one_photo():
    #img = cv2.imread("slike/hard/sudoku_hard_052.jpg", 0)
    #img = cv2.imread("slike/photos/5.jpg", 0)
    #img = inverte(img)
    img = transform_photo("slike/photos/5.jpg")
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
    tables = np.zeros((5265,784))
    counter = 0
    for f in range(1,61):
        s = "0"
        if f < 10:
            s = "00"
        img = cv2.imread("slike/easy/sudoku_easy_" + s + str(f) + ".jpg", 0)
        img = inverte(img)
        data = np.asarray(img, dtype="uint8")
        #numbers = [None]*81                
        for i in xrange(0,9):       #secemo tablu na 91 sliku
            for j in xrange(0,9):
                num = data[i*33:i*33+33, j*33:j*33+33]
                num = num[4:32, 4:32]
                num = num.reshape(1, 784)
                num = num.astype("float32")
                num /= 255
                tables[counter, :] = num
                counter += 1
        
        #for i in range(len(numbers)):
         #   num = numbers[i]
          #  num = num.reshape(1, 784)
           # num = num.astype("float32")
            #num /= 255
            #numbers[i] = num
        #tables[f-1] = numbers
       
    # ----- Ovo dalje moze da se zakomentarise
    for f in range(61,66):
        img = transform_photo("slike/photos/" + str(f - 60) + ".jpg")
        data = np.asarray(img, dtype="uint8")
        #numbers = [None]*81
        counter = 0  
        for i in xrange(0,9):       #secemo tablu na 91 sliku
            for j in xrange(0,9):
                num = data[i*33:i*33+33, j*33:j*33+33]
                num = num[4:32, 4:32]
                num = num.reshape(1, 784)
                num = num.astype("float32")
                num /= 255
                tables[f-1, :] = num
                counter += 1
        
        #for i in range(len(numbers)):
        #    num = numbers[i]
        #    num = num.reshape(1, 784)
        #    num = num.astype("float32")
        #    num /= 255
        #    numbers[i] = num
        #tables[f-1] = numbers
    return tables
   
def get_classes():
    tables = [None]*65
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
    
    for f in range(61, 66):
        file = open("slike/photos/" + str(f - 60) + ".dat", "r")
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
    
def get_size(biggest):
    width_gore = biggest[0][0][0] - biggest[1][0][0]
    width_dole = biggest[3][0][0] - biggest[2][0][0]
    height_levo = biggest[2][0][1] - biggest[1][0][1]
    height_desno = biggest[3][0][1] - biggest[0][0][1]
    width = width_gore if width_gore > width_dole else width_dole
    height = height_levo if height_levo > height_desno else height_desno
    return (width, height)

def transform_photo(path):
    print path
    img = cv2.imread(path, 0)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    img = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    img = cv2.adaptiveThreshold(img,255,1,1,11,4)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    _, contours, hierarchy = cv2.findContours(copy.deepcopy(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    for i in contours:
            area = cv2.contourArea(i)
            if area > img.size/4:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                    
    # Prikaz pronadjenih uglova table
    #cv2.circle(img,(biggest[0][0][0], biggest[0][0][1]), 10, (255,0,0), 1) # gore desno
    #cv2.circle(img,(biggest[1][0][0], biggest[1][0][1]), 10, (255,0,0), 1) # gore levo
    #cv2.circle(img,(biggest[2][0][0], biggest[2][0][1]), 10, (255,0,0), 1) # dole levo
    #cv2.circle(img,(biggest[3][0][0], biggest[3][0][1]), 10, (255,0,0), 1) # dole desno
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    s_width, s_height = get_size(biggest)
   # print (biggest[0][0])
    pts1 = np.float32([biggest[1][0], biggest[0][0], biggest[2][0], biggest[3][0]])
    pts2 = np.float32([[0,0],[s_width, 0],[0, s_height],[s_width, s_height]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    img = cv2.warpPerspective(img,M,(s_width,s_height))
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return img