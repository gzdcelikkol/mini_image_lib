import numpy as np
from core.conv import pad_image

def get_structuring_element(size: int = 3, shape: str = "rect") -> np.ndarray:
    """Morfolojik işlemler için kernel üretir."""
    if shape == "rect":
        return np.ones((size, size), dtype=np.uint8)
    elif shape == "cross":
        kernel = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        kernel[mid, :] = 1
        kernel[:, mid] = 1
        return kernel
    else:
        raise ValueError("Desteklenen şekiller: 'rect', 'cross'")

def erode(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Görüntüyü aşındırır (Erosion).
    Kernel altındaki piksellerin minimumunu alır.
    """
    if kernel is None:
        kernel = get_structuring_element(3, "rect")
        
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    pad_h, pad_w = k_h // 2, k_w // 2

    # Kenarlarda min alırken yapay sınır oluşmaması için kenarları yüksek değerle doldurur
    padded = pad_image(image, pad_h, pad_w, mode="edge")
    
    output = np.zeros_like(image)
    kernel_mask = kernel.astype(bool)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i : i + k_h, j : j + k_w]
            # Kernelin '1' olduğu yerlerdeki piksellerin minimumu
            output[i, j] = np.min(region[kernel_mask])
            
    return output

def dilate(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Görüntüyü genişletir (Dilation).
    Kernel altındaki piksellerin maksimumunu alır.
    """
    if kernel is None:
        kernel = get_structuring_element(3, "rect")
        
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = pad_image(image, pad_h, pad_w, mode="edge")
    
    output = np.zeros_like(image)
    kernel_mask = kernel.astype(bool)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i : i + k_h, j : j + k_w]
            # Kernelin '1' olduğu yerlerdeki piksellerin maksimumu
            output[i, j] = np.max(region[kernel_mask])
            
    return output

def opening(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """Açma: Önce Erosion, sonra Dilation."""
    return dilate(erode(image, kernel), kernel)

def closing(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """Kapama: Önce Dilation, sonra Erosion."""
    return erode(dilate(image, kernel), kernel)