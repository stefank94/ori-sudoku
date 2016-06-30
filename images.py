import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


def get_one_photo():
    #img = cv2.imread("slike/hard/sudoku_hard_052.jpg", 0)
    #img = cv2.imread("slike/photos/5.jpg", 0)
    #img = inverte(img)
    img = transform_photo("slike/photos/9.jpg")
    data = np.asarray(img, dtype="uint8")
    numbers = [None]*81
    counter = 0        
    for i in xrange(0,9):       #secemo tablu na 91 sliku
        for j in xrange(0,9):
            num = data[i*33:i*33+33, j*33:j*33+33]
            num = num[4:32, 4:32]
            numbers[counter] = num
            counter += 1
    for i in range(81):
        plt.subplot(9,9,i+1)
        plt.imshow(numbers[i], cmap='gray', interpolation='none') #stampa celu tablu
    
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
    for i in range(81):
        plt.subplot(9,9,i+1)
        plt.imshow(numbers[i], cmap='gray', interpolation='none') #stampa celu tablu
    
    for i in range(81):
        num = numbers[i]
        num = num.reshape(1, 784)
        num = num.astype("float32")
        num /= 255
        numbers[i] = num
    return numbers
        
def get_images_for_learning():
    #tables = np.zeros((5265,784))
    tables = np.zeros((8910,784))
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
        for i in xrange(0,9):       #secemo tablu na 91 sliku
            for j in xrange(0,9):
                num = data[i*33:i*33+33, j*33:j*33+33]
                num = num[4:32, 4:32]
                num = num.reshape(1, 784)
                num = num.astype("float32")
                num /= 255
                tables[counter, :] = num
                counter += 1
       
    for f in range(66,76):
        img = transform_photo("slike/photos/1/" + str(f - 65) + ".jpg")
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
    for f in range(76,83):
        img = transform_photo("slike/photos/2/" + str(f - 75) + ".jpg")
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
    """for f in range(83,89):
        img = transform_photo("slike/photos/3/" + str(f - 82) + ".jpg")
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
                counter += 1"""
    for f in range(83,87):
        img = transform_photo("slike/photos/4/" + str(f - 82) + ".jpg")
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
    for f in range(87,92):
        img = transform_photo("slike/photos/5/" + str(f - 86) + ".jpg")
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
    for f in range(92,103):
        img = transform_photo("slike/photos/6/" + str(f - 91) + ".jpg")
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
    for f in range(103,111):
        img = transform_photo("slike/photos/7/" + str(f - 102) + ".jpg")
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
    #tables = [None]*65
    tables = [None]*110
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
    
    for f in range(66, 76):
        file = open("slike/photos/1/1.dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    for f in range(76, 83):
        file = open("slike/photos/2/2.dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    """
    for f in range(83, 89):
        file = open("slike/photos/3/" + str(f - 82) + ".dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    """
    for f in range(83, 87):
        file = open("slike/photos/4/4.dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    for f in range(87, 92):
        file = open("slike/photos/5/5.dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    for f in range(92, 103):
        file = open("slike/photos/6/6.dat", "r")
        list = [None]*81
        j = 0
        for line in file:
            chars = line.strip().split(' ')
            for char in chars:
                x = float(char)
                list[j] = x
                j += 1   
        tables[f-1]=list
    for f in range(103, 111):
        file = open("slike/photos/7/7.dat", "r")
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
    
def comp(crit,najmanji,x1,x2):
    if crit == "w" and najmanji == True:
        if x1[0] > x2[0]:
            return True
        return False
    elif crit == "w" and najmanji == False:
        if x1[0] < x2[0]:
            return True
        return False
    elif crit == "h" and najmanji == True:
        if x1[1] > x2[1]:
            return True
        return False
    elif crit == "h" and najmanji == False:
        if x1[1] < x2[1]:
            return True
        return False
    else:
        return None

def najmanji(crit,najmanji,lista):
    swapped = True
    while swapped:
        swapped = False
        for i in range(1,len(lista)):
            if comp(crit,najmanji,lista[i-1],lista[i]):
                lista[i-1], lista[i] = lista[i], lista[i-1]
                swapped = True
    return lista
    
def get_size(biggest):
    sorted_by_width_asc = najmanji("w", True, [ biggest[0][0], biggest[1][0], biggest[2][0], biggest[3][0] ])
    sorted_by_height_asc = najmanji("h", True, [ biggest[0][0], biggest[1][0], biggest[2][0], biggest[3][0] ])
    gl = sorted_by_width_asc[3] if sorted_by_width_asc[3][1] < sorted_by_width_asc[2][1] else sorted_by_width_asc[2]
    dl = sorted_by_width_asc[3] if sorted_by_width_asc[3][1] > sorted_by_width_asc[2][1] else sorted_by_width_asc[2]
    gd = sorted_by_width_asc[0] if sorted_by_width_asc[0][1] < sorted_by_width_asc[1][1] else sorted_by_width_asc[1]
    dd = sorted_by_width_asc[0] if sorted_by_width_asc[0][1] > sorted_by_width_asc[1][1] else sorted_by_width_asc[1]
    #width_gore = abs(biggest[0][0][0] - biggest[1][0][0])
    #width_dole = abs(biggest[3][0][0] - biggest[2][0][0])
    #height_levo = abs(biggest[2][0][1] - biggest[1][0][1])
    #height_desno = abs(biggest[3][0][1] - biggest[0][0][1])
    width_gore = abs(gd[0] - gl[0])
    width_dole = abs(dd[0] - dl[0])
    height_levo = abs(dl[1] - gl[1])
    height_desno = abs(dd[1] - gd[1])
    width = width_gore if width_gore > width_dole else width_dole
    height = height_levo if height_levo > height_desno else height_desno
    return ([gl,gd,dl,dd],width, height)

def transform_photo(path):
    #print path
    img = cv2.imread(path, 0)
    
    h, w = img.shape
    ratio =  h / w
    img= cv2.resize(img,(500, 500 * ratio),interpolation=cv2.INTER_AREA)
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
    if biggest is None:
        print("nooo " +  path)
    lista, s_width, s_height = get_size(biggest)
   # print (biggest[0][0])
    pts1 = np.float32([lista[1], lista[0], lista[3], lista[2]])
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