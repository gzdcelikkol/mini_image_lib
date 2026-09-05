import numpy as np
from PIL import Image
import os

from core import (
    convolve2d,
    gaussian_kernel,
    sobel_filters,
    simple_canny,
    get_structuring_element,
    erode,
    dilate,
    opening,
    closing,
    fft2d,
    fftshift,
    apply_frequency_filter
)
from utils import plot_before_after, plot_multiple

def create_synthetic_test_image(size: int = 128) -> np.ndarray:
    """
    Dışarıdan bir resim gerekmeden test yapabilmek için 
    kare, daire ve çizgilerden oluşan sentetik bir gri tonlamalı resim üretir.
    Boyut FFT algoritması için 2'nin kuvveti (örn: 128x128) olarak seçilmiştir.
    """
    img = np.zeros((size, size), dtype=np.float64)
    
    # Ortaya kare çiz
    img[size//4 : 3*size//4, size//4 : 3*size//4] = 180.0
    
    # İçine daire çiz
    y, x = np.ogrid[:size, :size]
    mask = (x - size//2)**2 + (y - size//2)**2 <= (size//6)**2
    img[mask] = 255.0
    
    # Birkaç ince çizgi ekle
    img[10:15, :] = 200.0
    img[:, 10:15] = 200.0
    
    return img

def load_or_create_image(image_path: str = None, target_size: int = 128) -> np.ndarray:
    """Belirtilen resim varsa onu okur ve griye çevirir; yoksa kendisi üretir."""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((target_size, target_size))
        return np.array(img, dtype=np.float64)
    return create_synthetic_test_image(size=target_size)

def main():
    
    # Görüntü hazırlama:
    # Manuel görüntü eklemek için base_img = load_or_create_image("foto.png")
    base_img = load_or_create_image()
    
    # Konvolüsyon ve Gauss Bulanıklaştırma 
    print("1. Gauss Bulanıklaştırma test ediliyor...")
    g_kernel = gaussian_kernel(size=7, sigma=1.8)
    blurred = convolve2d(base_img, g_kernel)
    plot_before_after(base_img, blurred, "Orijinal Görüntü", "Gauss Bulanıklaştırma (7x7, sigma=1.8)")

    # Kenar Tespiti (Sobel ve Canny) 
    print("2. Kenar Tespiti (Sobel ve Canny) test ediliyor...")
    gx, gy, magnitude, _ = sobel_filters(base_img)
    canny_edges = simple_canny(base_img, low_threshold=40, high_threshold=100)
    
    plot_multiple(
        [base_img, np.abs(gx), np.abs(gy), magnitude, canny_edges],
        ["Orijinal", "Sobel Gx", "Sobel Gy", "Gradyan Büyüklüğü", "Basit Canny"]
    )

    # Morfolojik İşlemler (Erosion, Dilation, Opening, Closing) ---
    print("3. Morfolojik İşlemler:")

    # Test için biraz tuz-biber gürültüsü ve delikler eklenir
    noisy_img = base_img.copy()
    np.random.seed(42)
    noise_mask = np.random.rand(*base_img.shape) < 0.03
    noisy_img[noise_mask] = 255.0  # Tuz gürültüsü (beyaz noktalar)
    holes_mask = np.random.rand(*base_img.shape) < 0.02
    noisy_img[holes_mask] = 0.0    # Biber gürültüsü (siyah delikler)
    
    struct_elem = get_structuring_element(size=3, shape="rect")
    eroded = erode(noisy_img, struct_elem)
    dilated = dilate(noisy_img, struct_elem)
    opened = opening(noisy_img, struct_elem)   # Beyaz noktaları temizler
    closed = closing(noisy_img, struct_elem)   # Siyah delikleri kapatır
    
    plot_multiple(
        [noisy_img, eroded, dilated, opened, closed],
        ["Gürültülü", "Erosion (Aşındırma)", "Dilation (Genişletme)", "Opening (Açma)", "Closing (Kapama)"]
    )

    # 2D Fourier Dönüşümü ve Frekans Filtreleri 
    print("4. Fourier Dönüşümü ve Frekans Filtreleme test ediliyor...")
    # Frekans spektrumunu görselleştirelim (log magnitude)
    f_shift = fftshift(fft2d(base_img))
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    low_passed = apply_frequency_filter(base_img, cutoff=20, filter_type="low")
    high_passed = apply_frequency_filter(base_img, cutoff=20, filter_type="high")
    
    plot_multiple(
        [base_img, magnitude_spectrum, low_passed, high_passed],
        ["Orijinal", "FFT Büyüklük Spektrumu", "Low-Pass Filtre (Cutoff=20)", "High-Pass Filtre (Cutoff=20)"]
    )
    


if __name__ == "__main__":
    main()