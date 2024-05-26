"""
This script performs various image processing tasks using multiprocessing in Python. The tasks include:
1. Displaying the original image and its separate color channels (red, green, and blue).
2. Storing the green channel as a separate image file.
3. Calculating the mean value of the green channel.
4. Computing the fraction of pixels in the red channel with values smaller than 50.
5. Plotting the histogram of the red channel.
6. Detecting edges in the green channel using the Canny edge detection algorithm.

The multiprocessing module is utilized to parallelize these tasks, improving efficiency by executing them concurrently on multiple CPU cores. Each task is encapsulated within a separate process to run independently of others, enhancing overall performance when handling computationally intensive image processing operations.


This was individually coded by: 

Aryan jain, 22107593
Harsh Gurawaliya, 22109730
"""




import cv2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np 

img = cv2.imread('ukulele.jpg')
blue_channel, green_channel, red_channel = cv2.split(img)

def showcase(): 
    plt.figure(figsize=(10, 6))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Red Channel 
    plt.subplot(2, 2, 2)
    plt.imshow(red_channel, cmap='gray')
    plt.title('Red Channel')
    plt.axis('off')

    # Green Channel 
    plt.subplot(2, 2, 3)
    plt.imshow(green_channel, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')

    # Blue Channel
    plt.subplot(2, 2, 4)
    plt.imshow(blue_channel, cmap='gray')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def storeImage():
    cv2.imwrite('green_channel_image.jpg', green_channel, [cv2.IMWRITE_JPEG_QUALITY,100])

def mean_calc():
    green_mean = np.mean(green_channel)
    print("Mean value of the green channel:", green_mean)

def calc_frac_red_Channel(): 
    red_frac = red_channel < 50 
    pixels = np.sum(red_frac)
    total_pixels = red_channel.size
    fraction = pixels/total_pixels
    print("Fraction of pixels in the red channel with value smaller than 50:", fraction)

def calc_histogram(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color='red')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Red Channel')
    plt.show()

def edge_detection(channel):
    # The parameters 100 and 200 represent the lower and upper thresholds respectively for edge detection, using the Canny edge detection algorithm
    edges = cv2.Canny(channel, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges Detected in Green Channel')
    plt.axis('off')
    plt.show()
    
    pass
if __name__ == '__main__':
    pr1 = multiprocessing.Process(target=showcase)
    pr2 = multiprocessing.Process(target=storeImage)
    pr3 = multiprocessing.Process(target=mean_calc)
    pr4 = multiprocessing.Process(target = calc_frac_red_Channel)
    pr5 = multiprocessing.Process(target= calc_histogram, args=(red_channel,))
    pr6 = multiprocessing.Process(target=edge_detection, args=(green_channel,))

# Multiprocess threading 
# Multiprocess threading start
    pr1.start()
    pr2.start()
    pr3.start()
    pr4.start()
    pr5.start()
    pr6.start()

# Multiprocess threading Join
    pr1.join()
    pr2.join()
    pr3.join()
    pr4.join()
    pr5.join()
    pr6.join()


