import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np





def process_images(image):
    imagearray=mpimg.imread(image)
    return imagearray

#it turns out that we cannot process data using get_data functions even though I feel it is correct implementation for geting image data
#The reason is that the driving_log.csv contains image file names that don't exist in the IMG fold.
#Thus, I changed the get_data into precess_data functions
def get_data(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            directory = "../IMG"
            img_center = process_image(directory + row[0])
            img_left = process_image(directory + row[1])
            img_right = process_image(directory + row[2])

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)


