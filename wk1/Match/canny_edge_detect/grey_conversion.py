import numpy as np

def grey_transfer(image):
    #ITU-R BT.601 标准
    img = np.asarray(image, dtype=np.float32)
    img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    img = np.clip(img, 0,255).astype(np.uint8)
    return img