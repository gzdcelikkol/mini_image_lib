import numpy as np
from core.conv import convolve2d

def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Belirtilen boyut ve sigma değerinde 2D Gauss kernel üretir."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def sobel_filters(image: np.ndarray):
    """
    Yatay (Gx) ve dikey (Gy) Sobel ile çizgileri bulur ve 
    gradyan büyüklüğü ile yön açısını hesaplar.
    """
    # Sobel kernelleri 
    kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], dtype=np.float64)
                   
    ky = np.array([[-1, -2, -1], 
                   [ 0,  0,  0], 
                   [ 1,  2,  1]], dtype=np.float64)
    
    # convolve2d ile konvolüsyon uygulama
    gx = convolve2d(image, kx)
    gy = convolve2d(image, ky)
    
    # Gradyan büyüklüğü ve yönü
    magnitude = np.hypot(gx, gy)  # sqrt(gx^2 + gy^2) ile eşdeğerdir
    
    # 0 - 255 aralığına normalize et
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255.0
        
    theta = np.arctan2(gy, gx)
    return gx, gy, magnitude, theta

def simple_canny(image: np.ndarray, low_threshold: float = 30.0, high_threshold: float = 80.0) -> np.ndarray:
    """
    Basitleştirilmiş Canny kenar dedektörü:
    1. Gauss bulanıklaştırma
    2. Sobel gradyanı
    3. Çift eşikleme (Hysteresis/Double Thresholding)
    """
    
    # 1. Gürültüyü azaltmak için Gauss filtresi
    g_kernel = gaussian_kernel(size=5, sigma=1.2)
    smoothed = convolve2d(image, g_kernel)
    
    # 2. Gradyan büyüklüğünü bul
    _, _, magnitude, _ = sobel_filters(smoothed)
    
    # 3. Basit çift eşikleme
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    
    strong_i, strong_j = np.where(magnitude >= high_threshold)
    weak_i, weak_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))
    
    # Güçlü kenarlar kesinlikle kenardır (255)
    edges[strong_i, strong_j] = 255
    # Zayıf kenarlar (50) - opsiyonel takip için
    edges[weak_i, weak_j] = 50
    
    return edges