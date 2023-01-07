import sys
sys.path.insert(0, '/Users/trac.k.y/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cvlib')

import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly


image = cv2.imread("cars.jpg")
cube = cv2.imread("rubix cube.jpg")
#image2 = cv2.imread("cars copy.jpg")
#box, label, count = cv.detect_common_objects(image)
#output = draw_bbox(image, box, label, count)
'''
#matplot reads in RGB. convert from opencv's GBR to plot correct color image
plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(plt_image) 
plt.show()
#print("Number of cars in this image are " +str(label.count('car')))
'''

'''
#python's image reading array system:
#3d array (color image)
print(image[0]) #RGB values of the first row of pixels in the image
print(image[0, 0]) #RGB values of first pixel of first row (row, column coords)
print(image[0][0][0]) #R value of first pixel of first row

#adds a thicc black line on image
row = 0
while row < 900:
    column = 0
    while column < 100:
        image[row, column] = (0,0,0) #tracy learns array notation (:
        column += 1
    row += 1

#actual rectangle drawing function
counter = 100
while counter >= 0:
    cv2.rectangle(image, (0,0), (counter,counter), (255,0,0), -1)
    counter -= 1

#actual line drawing function
#i set this so it actually does the same thing as the first while loop
cv2.line(image, (50,0), (50, 900), (0, 0, 0), 100)

#flipping image upside down
start = 0
end = 899
while start < end:
    image[[start, end]] = image[[end, start]] #tracy learns array notation pt 2
    start += 1
    end -= 1

#flipping image left to right
start = 0
end = 1199
while start < end:
    image[: ,[start, end]] = image[: ,[end, start]] 
    start += 1
    end -= 1
'''

'''
#rotating 90 degrees
old = image
new = 255 * np.ones(shape=[1200,900,3],dtype='uint8') #initialize blank white image
#print(old)
#print(new)

row = 0
column = 899
while row < 900 and column >= 0:
    index = 0
    while index < 1200:
        new[index, column] = old[row, index]
        index += 1
    column -= 1
    row += 1

#rotate to 180
new2 = 255 * np.ones(shape=[900,1200,3],dtype='uint8')
row = 0
column = 1199
while row < 1200 and column >= 0:
    index = 0
    while index < 900:
        new2[index, column] = new[row, index]
        index += 1
    column -= 1
    row += 1

#rotate to 270
new3 = 255 * np.ones(shape=[1200,900,3],dtype='uint8')
row = 0
column = 899
while row < 900 and column >= 0:
    index = 0
    while index < 1200:
        new3[index, column] = new2[row, index]
        index += 1
    column -= 1
    row += 1
'''
'''
#function definition way of rotating
width = 1200
height = 900
rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)
print(rotationMatrix) #og matrix multiplied by this one gives the rotated matrix
rotatedImage = cv2.warpAffine(image, rotationMatrix, (width, height))
print(rotatedImage.shape)
#cv2.imshow("Rotated Image", rotatedImage)
'''

#personal function definition of rotation
def rot90(image):
    old = image
    height = image.shape[0]
    width = image.shape[1]
    new = 255 * np.ones(shape=[width,height,3],dtype='uint8')

    row = 0
    column = height - 1
    while row < height and column >= 0:
        index = 0
        while index < width:
            new[index, column] = old[row, index]
            index += 1
        column -= 1
        row += 1

    return new

'''
#using vectors
old = image
new = 255 * np.ones(shape=[1200,900,3],dtype='uint8') 
row = 0
column = 899
while row < 900 and column >= 0:
    new[:, column] = old[row]
    column -= 1
    row += 1
'''
'''
#display multiple images in multiple windows
cv2.namedWindow("1")
cv2.imshow("1", image)

cv2.namedWindow("2")
cv2.imshow("2", image2)

cv2.namedWindow("before")
cv2.imshow("before", old)

cv2.namedWindow("90")
cv2.imshow("90 right", new)

cv2.namedWindow("180")
cv2.imshow("180", new2)

cv2.namedWindow("270")
cv2.imshow("270", new3)
'''

#convert to grayscale
#grayscale image formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
#method 1: don't reduce dimensions of array.
#get the grayscale brightness value, then change r, g, b values to all be that value
#pixel version
'''
old = image
grayfactor = np.array([0.1140, 0.5870, 0.2989])
grayscale = image * grayfactor
for row in grayscale:
    for pixel in row:
        pixelsum = pixel[0] + pixel[1] + pixel[2]
        pixel[0] = pixelsum
        pixel[1] = pixelsum
        pixel[2] = pixelsum
grayscale = np.array(grayscale, dtype = 'uint8')
cv2.imshow("original", image)
cv2.imshow("grayscale", grayscale)
'''
'''
#vector version
old = image
grayfactor = np.array([0.1140, 0.5870, 0.2989])
grayscale = image * grayfactor
for row in grayscale:
    for pixel in row:
        pixelsum = pixel[0] + pixel[1] + pixel[2]
        pixel[0] = pixelsum
        pixel[1] = pixelsum
        pixel[2] = pixelsum
grayscale = np.array(grayscale, dtype = 'uint8')
cv2.imshow("original", image)
cv2.imshow("grayscale", grayscale)
'''
'''
#jk i figured out how to make a 3x1 array.
#heres method two where we also learn that matrix multiplication
#is actually not the same as the times sign
old = image
grayfactor = np.empty((3, 1))
grayfactor[0] = 0.1140
grayfactor[1] = 0.5870
grayfactor[2] = 0.2989
grayscale = image.dot(grayfactor)
grayscale = np.array(grayscale, dtype = 'uint8')
cv2.imshow("original", image)
cv2.imshow("grayscale", grayscale)
'''

def grayscale(image):
    grayfactor = np.empty((3, 1))
    grayfactor[0] = 0.1140
    grayfactor[1] = 0.5870
    grayfactor[2] = 0.2989
    grayscale = image.dot(grayfactor)
    grayscale = np.array(grayscale, dtype = 'uint8')
    return grayscale

def draw_vert_line(image, linewidth, color = [0, 0, 0]):
    row = 0
    while row < len(image[:, 0]):
        column = 0
        while column < linewidth:
            image[row, column] = color
            column += 1
        row += 1
    return image
'''
#incremental turn to grayscale, in same window
cv2.imshow("image", cube)
for i in range(0, cube.shape[0], 100):
    #print(image[i:i+99])
    cube[i:i+100] = grayscale(cube[i:i+100])
    #print(image[i:i+99])
    cv2.waitKey(1000)
    cv2.imshow("image", cube)
'''
#as a function
def grayscale_byrow(image, window = 'image', delay = 1000, width=100):
    cv2.imshow(window, image)
    for i in range(0, image.shape[0], width):
        image[i:i+width] = grayscale(image[i:i+width])
        cv2.waitKey(delay)
        cv2.imshow(window, image)

#grayscale_byrow(image, window = 'cars')
#grayscale_byrow(cube, window = 'cube', width = 500)

'''
overlaying images
formula: result = array1*alpha + array2*beta + gamma (gamma = 0)
alpha + beta = 1
'''
test1 = cv2.imread("test1.jpeg")
test2 = cv2.imread("test2.jpeg")


def image_overlay(image1, image2, opacity1 = 0.5):
    opacity2 = 1 - opacity1
    result = (opacity1 * image1) + (opacity2 * image2)
    result = np.array(result, dtype='uint8')
    cv2.imshow('result', result)
    
image_overlay(test1, test2)
#image_overlay(test1, test2, opacity1 = 0.2)
