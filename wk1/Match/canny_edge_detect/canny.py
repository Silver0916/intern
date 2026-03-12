import numpy as np
from grey_conversion import grey_transfer
from gaussian import gaussian_kernel, gaussian_blur
from collections import deque
from itertools import product
import argparse
import cv2


def conv(image, kernel, padding_mode='reflect', clip = True):
    img_size = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2 
    
    # padding
    img = np.asarray(image, dtype=np.float32)
    img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode=padding_mode)

    # convolution
    conv_img = np.zeros_like(image, dtype=np.float32)

    for x in range(img_size[0]):
        for y in range(img_size[1]):
            conv_img[x,y] = np.sum(img[x: x+kernel_size, y: y+kernel_size] * kernel)

    if clip:
        conv_img = np.clip(conv_img, 0, 255).astype(np.float32)

    return conv_img

def canny_edge_detection(image, low_thr, high_thr):
    # grey conversion
    grey_img = grey_transfer(image)

    # gaussian blur
    kernel = gaussian_kernel(kernel_size = 5, sigma = 1.0)
    blurred_img = gaussian_blur(grey_img, kernel, padding_mode='reflect')

    # sobel operator
    sobel_x = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]])
    
    grad_x = conv(blurred_img, sobel_x, padding_mode='reflect',clip = False)
    grad_y = conv(blurred_img, sobel_y, padding_mode='reflect',clip = False)
    grad = np.hypot(grad_x, grad_y)
    grad_dir = np.arctan2(grad_y, grad_x) * (180. / np.pi)
    grad_dir[grad_dir < 0] += 180

    # non-maximum suppression
    nms_img = np.zeros_like(grad, dtype =np.float32)
    for x in range(1, grad.shape[0]-1):
        for y in range(1, grad.shape[1]-1):
            if (0 <= grad_dir[x,y] <22.5):
                if (grad[x,y] < grad[x,y+1] and grad[x,y] < grad[x,y-1]):
                    nms_img[x,y] = 0 
                else:
                    nms_img[x,y] = grad[x,y]
            elif (22.5 <= grad_dir[x,y] < 67.5):
                if (grad[x,y] < grad[x-1,y+1] and grad[x,y] < grad[x+1,y-1]):
                    nms_img[x,y] = 0
                else:
                    nms_img[x,y] = grad[x,y]
            elif (67.5 <= grad_dir[x,y] < 112.5):
                if (grad[x,y] < grad[x-1,y] and grad[x,y] < grad[x+1,y]):
                    nms_img[x,y] = 0
                else:
                    nms_img[x,y] = grad[x,y]
            elif (112.5 <= grad_dir[x,y] < 157.5):
                if (grad[x,y] < grad[x-1,y-1] and grad[x,y] < grad[x+1,y+1]):
                    nms_img[x,y] = 0
                else:
                    nms_img[x,y] = grad[x,y]
            else:
                if (grad[x,y] < grad[x,y+1] and grad[x,y] < grad[x,y-1]):
                    nms_img[x,y] = 0
                else:
                    nms_img[x,y] = grad[x,y]
    
    # Double thresholding
    STRONG_VALUE = 255
    WEAK_VALUE = 75
    NONE_VALUE = 0
    strong = (nms_img >= high_thr)
    weak = ((nms_img >= low_thr) & (nms_img < high_thr))
    none = (nms_img < low_thr)
    out = nms_img.copy()
    out[strong] = STRONG_VALUE
    out[weak] = WEAK_VALUE
    out[none] = NONE_VALUE

    q = deque()
    strong_coords  =np.argwhere(out == STRONG_VALUE)
    neighbors = [(dx, dy) for dx, dy in product([-1,0,1], [-1,0,1]) if (dx,dy) != (0,0)]

    for coord in strong_coords:
        q.append(tuple(coord))
    while q:
        x,y = q.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (0 <= nx < out.shape[0]) and (0 <= ny < out.shape[1]) and (out[nx, ny] == WEAK_VALUE):
                out[nx, ny] = STRONG_VALUE
                q.append((nx, ny))

    out[out != STRONG_VALUE] = 0
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canny Edge Detection")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--low_threshold", type=float, default=50.0, help="Low threshold for hysteresis")
    parser.add_argument("--high_threshold", type=float, default=150.0, help="High threshold for hysteresis")
    args = parser.parse_args()


    input_image = cv2.imread(args.input_image)
    edges = canny_edge_detection(input_image, args.low_threshold, args.high_threshold)

    output_root = r'D:\projs\intern\canny_edge_detect\canny_test_data'
    output_path = f"{output_root}/canny_edges.png"
    
    cv2.imwrite(output_path, edges)
    print(f"Canny edge detection completed. Output saved to: {output_path}")