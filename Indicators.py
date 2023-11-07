"""
Indicators library

In this library, a set of indicator functions are defined in order to be called at the pre-processing stage of the
discovery algorithm training engine.

All the functions should receive a chunk of size LxNxM and return just a 2D tensor (N, M). The main idea is collapsing
the time-dimension in order to feed the result to a discovery algorithm.
"""
import numpy as np
from scipy.stats import kurtosis, skew

def mean (array):
    """
    Calculate the mean of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the mean optical signal value for each pixel.
    """
    return np.mean(array, axis=0)

def median (array):
    """
    Calculate the median of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis..

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the median optical signal value for each pixel.
    """
    return np.median(array, axis=0)

def var(array):
    """
    Calculate the moment of third order of optical signals for multiple pixels.

   Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the third moment values.
    """
    
    return np.var(array, axis=0)

def std(array):
    """
    Calculate the standard deviation of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the standard deviation of optical signals 
        for each pixel.
    """
    return np.std(array, axis=0)

def moment_3(array):
    """
    Calculate the moment of third order of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the third moment values.
    """
    # Calculate the mean (first moment) along axis=2 for each pixel
    mean = np.mean(array, axis=0, keepdims=True)

    # Calculate the third-order moment for each pixel
    moment_3 = np.mean((array - mean)**3, axis=0)

    return moment_3

def max_peek_dif(array):
    """
    Calculate the maximum peak difference of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the maximum peak difference of optical 
        signals for each pixel.
    """
    return (np.max(array, axis=0) - np.min(array, axis=0))

def SNR(array):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the SNR of optical signals for each pixel.
    """
    # Calculate the SNR for each pixel
    num_pixels = array.shape[2]  # Get the number of pixels (M)
    snr_values = np.zeros((num_pixels, 1), dtype=np.float32)

    for pixel_idx in range(num_pixels):
        # Extract the pixel's data over time
        pixel_data = array[:, :, pixel_idx]
        
        # Calculate the mean signal and the standard deviation of noise
        mean_signal = np.mean(pixel_data, axis=0)
        noise = pixel_data - mean_signal
        std_noise = np.std(noise, axis=0)
        
        # Calculate SNR for the pixel
        snr_values[pixel_idx, 0] = np.mean(mean_signal) / np.mean(std_noise)

    return snr_values


def kurtosis(array):
    """
    Calculate the kurtosis of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the kurtosis values.
    """
    kurtosis = kurtosis(array, axis=0) 

    return kurtosis

def skewness(array):
    """
    Calculate the skewness of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the skewness values.
    """
    skewness = skew(array, axis=0)

    return skewness

def norme(array):
    """
    Calculate the norm (magnitude) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the norm values of the vectors.
    """
    norm_values = np.linalg.norm(array, axis=0)
    
    return norm_values

def percentil(array, percentil):
    """
    Calculate the specified percentile of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the percentil values of the vectors.
    """
    
    percentile_values = np.percentile(array, percentil, axis=0)

    return percentile_values
    

def standardization(array):
    """
    Calculates the z-score for each pixel in the last image of the sequence.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A NumPy array with the z-score standardization of the pixels in the last image.
    """ 
    last_image = array[-1, ...]
    mean_value = np.mean(array)
    std_value = np.std(array)

    return (last_image - mean_value) / (std_value + 1e-10)