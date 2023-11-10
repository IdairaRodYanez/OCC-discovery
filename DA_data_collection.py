"""
Transmitter vs. Non-Transmitter Discriminant Analysis Data Collection Script

This script is designed for getting the essential data for the analysis of various parameters that can 
aid in the differentiation of pixels associated with transmitters from other pixels. It focuses on 
examining specific sets of pixels across a 4-second segment of multiple videos. The collected data is 
stored in a CSV file for subsequent processing and in-depth analysis.

The primary objective of this script is to gather essential information necessary for investigating 
which parameters exhibit the most significant distinctions between a transmitter pixel and others.
"""

import numpy as np
import pandas as pd
import cv2 as cv
from scipy.stats import kurtosis, skew
import math
import time
import csv
import os
import inspect
import Indicators
import pickle


def get_init_df(module, file):  
    """
    Create an initial DataFrame with columns for functions in a module or get the one created before.

    Args:
        module: The Python module to inspect for functions.
        file: name of the file from which DataFrame will be recovered.

    Returns:
        pandas.DataFrame: A DataFrame with columns for each function and their associated "_time" columns.
    """
    if not os.path.exists(file):
        # Get all members of the module
        members = inspect.getmembers(module)

        # Filter only functions
        funtions = [name for name, object in members if inspect.isfunction(object)]

        df = pd.DataFrame()

        # Add columns to DataFrame
        for name in funtions:
            df[name] = []
        
        df['class'] = []
        print("file created")
    else:
        # Read pickle file as DataFrame
        df = pd.read_pickle(file)
        print("file reload")

    return df   


def extract_info(txt_file_directory):
    """
    Extract information from a text file containing pixel positions and FPS data.

    Args:
        txt_file_directory (str): The directory and filename of the text file.

    Returns:
        transmitters (list): A list of (x, y) coordinates of selected transmitters.
        non_transmitters (list): A list of (x, y) coordinates of selected non-transmitters.
        fps (float): Frames per second of the video.
    """
    transmitters = []
    fps = None
    
    with open(txt_file_directory, 'r') as file:
        # Read line by line
        for line in file:
            # Divide line in parts based on ,
            parts = line.strip().split(', ')
            if parts[0] == "T":
                    transmitters.append((int(parts[1]), int(parts[2])))
            elif parts[0] == "FPS":
                    fps = float(parts[1])
    
    return transmitters, fps

def prepare_opticals_signals(images, pixels, fps):
    """
    Extract optical signals for selected pixels from a series of images.

    Args:
        images (list): A list of grayscale images, where each image is a 2D array.
        pixels (list): A list of (x, y) pixel coordinates to select.

    Returns:
        np.ndarray: A 3D array of shape (60, 1, M), where M is the number of selected pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.
    """
    num_frames = len(images)
    num_pixels = len(pixels)
    
    # Initialize the result array with shape (num_frames, 1, num_pixels)
    optical_signals = np.zeros((num_frames, 1, num_pixels), dtype=np.float32)
    
    for frame_idx, image in enumerate(images):
        for pixel_idx, (x, y) in enumerate(pixels):
            optical_signals[frame_idx, 0, pixel_idx] = image[y, x]
    
    return optical_signals


def calculate_all(df, array, case):
    """
    Execute each function and count each time. 

    Store each value on a variable.

    Create a Pandas file with all the variables. 

    Store the information in a CSV file.
    """
    # Get all members of the module
    members = inspect.getmembers(Indicators)

    # Filter only functions
    functions = [(name, obj) for name, obj in members if inspect.isfunction(obj) and obj.__module__ == Indicators.__name__]

    # Create a temporary DataFrame to store new data
    new_data = pd.DataFrame()
    
    # Create times dictionary
    times = {}

    # Execute the function and assign the result to the corresponding column
    for function_name, function_callable in functions:

        # Excute function and assign result
        time_i = time.time()
        # Get columns length
        result = function_callable(np.array(array)).ravel()
        num_elements = result.size

        # Get processing time
        time_funtion =  time.time() - time_i
        # Assign result to column
        new_data[function_name] = result

        # Store times values in a dictionary
        times[function_name] = [time_funtion]  
    
    # Add a 'class' column
    class_array = np.full(num_elements, case)
    new_data['class'] = class_array

    # Concatenate the temporary DataFrame with the main DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

    columns_to_drop = ['ku', 'sk']
    columns_exist = all(col in df.columns for col in columns_to_drop)

    if columns_exist:
        df = df.drop(columns=columns_to_drop)
    
    return df, times
   


def main():
    """
    Access each video directory and get pixels for analysis.

    Obtain the first 180 images from each video and convert them to gray scale.

    Get optical signal of each pixel in 60 images segment.

    Combine the results into a single matrix.
    
    Calculate parameters for each segment of the video based on the selected pixels.

    Store the information in a CSV file.
    """

    ######################################## CHANGE ONLY THIS ##############################################
    case = "tx" # tx or notx (group of images that are being analyzed) 
    ######################################## CHANGE ONLY THIS ##############################################
    
    # Loop through directories
    for i in range(1, 2):
        try:
            # Get info of fps and transmitters positions
            transmitters, fps = extract_info(f"data/{i}/info.txt")
            print(transmitters)
            images = []

            # A pkl file for each video (includes tx and non tx info)
            pkl_name = f"data/{i}/DataColletion.pkl"
            # Create initial DataFrame with empty columns or reload the one stored
            df = get_init_df(Indicators, pkl_name)

            # Random init image for each directory (videos always start with the header of the data frame)
            init = int(np.random.randint(0, 2 * fps))
            steps = int(math.trunc(fps/15))
            end = int(init + 2*fps + 1)
            # Extract and convert all images in a data frame to grayscale
            for count in range(init, end, steps): # max 2 images per bit
                image = cv.imread("data/%d/%s/frame_%d.jpg" % (i,case, count))
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)                     
                images.append(gray) 
            
            if case == "tx":
                # Get optical signals of interest
                tx_array = prepare_opticals_signals(images, transmitters, fps)

                # Calculate parameters
                df, times = calculate_all(df, tx_array, "T") 

            elif case == "notx":
                # Calculate parameters
                df, times = calculate_all(df, images, "NT")

                # Save times dictionary to a txt file
                with open(f"data/{i}/times.txt", 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for key, value in times.items():
                        csvwriter.writerow([key, value])
            
            print(df)          
            # Save the DataFrames to a pkl file
            df.to_pickle(pkl_name)

            
        except (FileNotFoundError, ZeroDivisionError):
            print(f'video {i} no existe')
    

if __name__ == "__main__":
    main()
