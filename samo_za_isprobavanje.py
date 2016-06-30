import numpy as np
import cv2
import copy
import math

def get_size(biggest):
    width_gore = biggest[0][0][0] - biggest[1][0][0]
    width_dole = biggest[3][0][0] - biggest[2][0][0]
    height_levo = biggest[2][0][1] - biggest[1][0][1]
    height_desno = biggest[3][0][1] - biggest[0][0][1]
    width = width_gore if width_gore > width_dole else width_dole
    height = height_levo if height_levo > height_desno else height_desno
    return (width, height)
    
def find_index(lines, l):
    for i in range(len(lines)):
        one = lines[i]
        if one[0][0] == l[0][0] and one[0][1] == l[0][1] and one[0][2] == l[0][2]:
            return i
    return None
    
# --------------------------------------------------------------

img = cv2.imread('slike/photos/1.jpg', 0)
#img = cv2.imread('C:/Users/Stefan/Desktop/2.jpg', 0)
#img = cv2.imread("slike/hard/sudoku_hard_010.jpg", 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.adaptiveThreshold(img,255,1,1,11,4)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
print (biggest[0][0])
pts1 = np.float32([biggest[1][0], biggest[0][0], biggest[2][0], biggest[3][0]])
pts2 = np.float32([[0,0],[s_width, 0],[0, s_height],[s_width, s_height]])

M = cv2.getPerspectiveTransform(pts1,pts2)

img = cv2.warpPerspective(img,M,(s_width,s_height))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blank_image = np.ones((300,300,3), np.uint8)


edges = cv2.Canny(img,50 ,150,apertureSize = 5)

minLineLength=280
lines = cv2.HoughLinesP(image=edges,rho=0.08,theta=np.pi/500, threshold=6,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

to_delete = []
compared = []

for i in range(len(lines)):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    for j in range(len(lines)):
        if i == j:
            continue
        if [i, j] in compared:
            continue
        x3 = lines[j][0][0]
        y3 = lines[j][0][1]
        x4 = lines[j][0][2]
        y4 = lines[j][0][3]
        
        case = 3
        if y1==y2 and y3==y4: # Horizontal Lines
            diff = abs(y1-y3)
            case = 1
        elif x1==x2 and x3==x4: # Vertical Lines
            diff = abs(x1-x3)
            case = 2
        else:
            diff = 0
            case = 3
        

        if diff < 5 and diff != 0:
            if x1 == 4 and y1 == 3 and x2 == 299 and y2 == 3:
                print "BBB"
                #cv2.line(blank_image, (x3,y3), (x4,y4), (0,255,0), 1, cv2.LINE_AA)
            if x3 == 4 and y3 == 3 and x4 == 299 and y4 == 3:
                print [x1,y1,x2,y2]
                print "CCC"
                print case
                #cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
            to_delete.append(lines[j])
            if case == 1:
                new_y = (y1 + y3) / 2
                lines[i,0,:] = [x1, new_y, x2, new_y]
            elif case == 2:
                new_x = (x1 + x3) / 2
                lines[i,0,:] = [new_x, y1, new_x, y2]
            #cv2.line(blank_image, (lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
        compared.append([j, i])
        

       
        
cv2.imshow('image', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()       
#print (len(to_delete))
for i in to_delete:
    index = find_index(lines, i)
    if index is None:
        continue
    lines = np.delete(lines, index, 0)

a,b,c = lines.shape
for i in range(a):
    cv2.line(blank_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 1, cv2.LINE_AA)
#cv2.line(blank_image, (4, 3), (299, 3), (0,255,0), 1, cv2.LINE_AA)
cv2.imshow('image', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





