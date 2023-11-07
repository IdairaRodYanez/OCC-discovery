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

def mean (array):
    """
    Calculate the mean of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60), where M is the number of pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the mean optical signal value for each pixel.
    """

def std(array):
    """
    Calculate the standard deviation of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60), where M is the number of pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the standard deviation of optical signals 
        for each pixel.
    """

def max_peek_dif(array):
    """
    Calculate the maximum peak difference of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60), where M is the number of pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the maximum peak difference of optical 
        signals for each pixel.
    """

def SNR(array):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60), where M is the number of pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the SNR of optical signals for each pixel.
    """

def max_correlation(array, FPS):...


def kurtosis(array):
    """
    Calculate the kurtosis of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60) where M is the number of pixels,
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the kurtosis values.
    """

def norme(array):
    """
    Calculate the norm (magnitude) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60) where M is the number of pixels,
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the norm values of the vectors.
    """
    
def skewness(array):
    """
    Calculate the skewness of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60) where M is the number of pixels,
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the skewness values.
    """

def moment_3(array):
    """
    Calculate the moment of third order of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A 3D array of shape (M, 1, 60) where M is the number of pixels,
        1 represents a single channel, and 60 represents the optical signals in 60 frames.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the third moment values.
    """

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

def prepare_opticals_signals(images, pixels):
    """
    Extract optical signals for selected pixels from a series of images.

    Args:
        images (list): A list of grayscale images, where each image is a 2D array.
        pixels (list): A list of (x, y) pixel coordinates to select.

    Returns:
        np.ndarray: A 3D array of shape (M, 1, 60), where M is the number of selected pixels, 
        1 represents a single channel, and 60 represents the optical signals in 60 frames.
    """

def main():
    """
    Access each video directory and get pixels for analysis.

    Obtain the first 180 images from each video and convert them to gray scale.

    Get optical signal of each pixel in 60 images segment.

    Combine the results into a single matrix.
    
    Calculate parameters for each segment of the video based on the selected pixels.

    Store the information in a CSV file.
    """

if __name__ == "__main__":
    main()
    













