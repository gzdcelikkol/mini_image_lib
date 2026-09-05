import numpy as np

def pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str = "reflect") -> np.ndarray:
    """Görüntünün kenarlarına padding ekler."""
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Konvolüsyon işlemi uygulamak için fonksiyon
    """
    # Görüntü ve kernel boyutları
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # Kernel'i 180 derece çevir 
    kernel_flipped = np.flip(np.flip(kernel, 0), 1)
    
    # Kernel merkezini korumak için gereken padding ekle
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    padded = pad_image(image, pad_h, pad_w, mode="reflect")
    output = np.zeros_like(image, dtype=np.float64)
    
    # Bir sonraki piksele kaydır
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i : i + k_h, j : j + k_w]
            output[i, j] = np.sum(region * kernel_flipped)
            
    return output