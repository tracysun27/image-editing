import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.lib.polynomial import poly

cars = cv2.imread("cars.jpg")
cube = cv2.imread("rubix cube.jpg")
#line/rectangle drawing function
def draw_line(image):
    row = 0
    while row < len(image[0]):
        column = 0
        while column < 100:
            image[row, column] = (0,0,0)
            column += 1
        row += 1
    return image

#flipping image upside down
def flip_vertical(image):
    start = 0
    end = 899
    while start < end:
        image[[start, end]] = image[[end, start]]
        start += 1
        end -= 1
    return image

#flipping image left to right
def flip_horizontal(image):
    start = 0
    end = 1199
    while start < end:
        image[: ,[start, end]] = image[: ,[end, start]] 
        start += 1
        end -= 1
    return image

#rotate image 90 degrees to the right
def rot90(image):
    height = image.shape[0]
    width = image.shape[1]
    new = 255 * np.ones(shape=[width,height,3],dtype='uint8')

    row = 0
    column = height - 1
    while row < height and column >= 0:
        index = 0
        while index < width:
            new[index, column] = image[row, index]
            index += 1
        column -= 1
        row += 1

    return new
'''
#using vectors
old = cube
new = 255 * np.ones(shape=[1200,900,3],dtype='uint8') 
row = 0
column = 899
while row < 900 and column >= 0:
    new[:, column] = old[row]
    column -= 1
    row += 1
'''


#convert to grayscale
def grayscale(image):
    grayfactor = np.empty((3, 1))
    grayfactor[0] = 0.1140
    grayfactor[1] = 0.5870
    grayfactor[2] = 0.2989
    grayscale = image.dot(grayfactor)
    grayscale = np.array(grayscale, dtype = 'uint8')
    return grayscale

def grayscale_byrow(image, window = 'image', delay = 1000, width=100):
    cv2.imshow(window, image)
    for i in range(0, image.shape[0], width):
        image[i:i+width] = grayscale(image[i:i+width])
        cv2.waitKey(delay)
        cv2.imshow(window, image)

def image_overlay(image1, image2, opacity1 = 0.5):
    opacity2 = 1 - opacity1
    result = (opacity1 * image1) + (opacity2 * image2)
    result = np.array(result, dtype='uint8')
    cv2.imshow('result', result)

#cv2.imshow('cars',cars)

input(">>> Enter a picture file name.")
filename = input(">>> ")
try:
    picture = cv2.imread(filename)
    cv2.imshow(filename, picture)
except:
    print("Error in processing image.")
    print("Your file name may be invalid or in the wrong directory.")
    print("Or maybe your image is not in the correct format.")
    print("Unfortunately I am too lazy to code out this bit properly for now,")
    print("so this is what you get lmao.")
    print("Please try again.")
print(">>> Choose an option.")
print("Press 1 to rotate")
print("Press 2 to convert to grayscale")
print("Press Q to quit program")
choice = input(">>> ")
while True:
    if choice == "1":
        cv2.imshow("before", picture)
        rotated = rot90(picture)
        cv2.imshow("after", rotated)
    elif choice == "2":
        cv2.imshow("before", picture)
        grayscale = grayscale(picture)
        cv2.imshow("after", grayscale)
    else:
        break
    break
