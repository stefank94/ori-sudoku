import numpy as np
import cv2

img = cv2.imread('slike/easy/sudoku_easy_001.jpg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()