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
    list = []
    for i in range(od, do):
        s = "0"
        if od < 10:
            s = "00"
        img = cv2.imread("slike/easy/sudoku_easy_" + s + str(i) + ".jpg", 0)
        list.append(img)
        img = cv2.imread("slike/medium/sudoku_medium_" + s + str(i) + ".jpg", 0)
        list.append(img)
        img = cv2.imread("slike/hard/sudoku_hard_" + s + str(i) + ".jpg", 0)
        list.append(img)
    return list

def load_y_set(od, do):
    list = []
    for i in range(od, do):
        s = "0"
        if od < 10:
            s = "00"
        file = open("slike/easy/sudoku_easy_" + s + str(i) + ".dat", "read")
        one_file = []
        for line in file:
            one_row = []
            chars = line.strip().split(' ')
            for char in chars:
                x = int(char)
                one_row.append(x)
            one_file.append(one_row)
        list.append(one_file)

        file = open("slike/medium/sudokumedium_" + s + str(i) + ".dat", "read")
        one_file = []
        for line in file:
            one_row = []
            chars = line.strip().split(' ')
            for char in chars:
                x = int(char)
                one_row.append(x)
            one_file.append(one_row)
        list.append(one_file)

        file = open("slike/hard/sudoku_hard_" + s + str(i) + ".dat", "read")
        one_file = []
        for line in file:
            one_row = []
            chars = line.strip().split(' ')
            for char in chars:
                x = int(char)
                one_row.append(x)
            one_file.append(one_row)
        list.append(one_file)

    return list
