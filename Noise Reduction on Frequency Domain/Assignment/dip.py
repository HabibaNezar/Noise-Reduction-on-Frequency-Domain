import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image (ensure the path is correct)
image_path = r'C:\Users\olabe\Downloads\WhatsApp Image 2024-12-16 at 12.57.45 PM.jpeg'  # Replace with the full path to your image if necessary
image = cv2.imread(image_path, 0)

if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found. Please check the file path.")

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_shift) + 1)

# Add Gaussian noise to the image
noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Function to apply filters in frequency domain
def apply_filter(filter_mask, f_shift):
    filtered = f_shift * filter_mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

# Get image dimensions
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Define Filters
# Low-Pass Filter
low_pass = np.zeros((rows, cols), np.uint8)
low_pass[crow-30:crow+30, ccol-30:ccol+30] = 1

# High-Pass Filter
high_pass = np.ones((rows, cols), np.uint8)
high_pass[crow-30:crow+30, ccol-30:ccol+30] = 0

# Band-Pass Filter
band_pass = np.zeros((rows, cols), np.uint8)
band_pass[crow-60:crow+60, ccol-60:ccol+60] = 1
band_pass[crow-30:crow+30, ccol-30:ccol+30] = 0

# Apply Filters
low_pass_result = apply_filter(low_pass, f_shift)
high_pass_result = apply_filter(high_pass, f_shift)
band_pass_result = apply_filter(band_pass, f_shift)

# Display Results
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Fourier Spectrum')
plt.subplot(233), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(234), plt.imshow(low_pass_result, cmap='gray'), plt.title('Low Pass Filter')
plt.subplot(235), plt.imshow(high_pass_result, cmap='gray'), plt.title('High Pass Filter')
plt.subplot(236), plt.imshow(band_pass_result, cmap='gray'), plt.title('Band Pass Filter')
plt.tight_layout()
plt.show()