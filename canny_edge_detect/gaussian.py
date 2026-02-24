import numpy as np

def gaussian_kernel(kernel_size, sigma):

    assert kernel_size % 2  == 1, "Kernel size must be odd."

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # kernel value
    for x in range(-center,  center+1):
        for y in range(-center, center+1):
            kernel[x + center,y + center] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)# normalize
    return kernel

def gaussian_blur(image, kernel, padding_mode='reflect'):

    img_size = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2 
    
    # padding
    img = np.asarray(image, dtype=np.float32)
    img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode=padding_mode)

    # convolution
    blurred_img = np.zeros_like(image, dtype=np.float32)

    for x in range(img_size[0]):
        for y in range(img_size[1]):
            blurred_img[x,y] = np.sum(img[x: x+kernel_size, y: y+kernel_size] * kernel)
    
    blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)

    return blurred_img