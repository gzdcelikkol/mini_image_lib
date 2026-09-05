import numpy as np

def fft1d(x: np.ndarray) -> np.ndarray:
    """
    Dizi uzunluğu 2'nin kuvveti olmalıdır. 2'nin kuvveti değilse sıfırdan DFT'ye fallback yapar.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    
    # Boyut 2'nin kuvveti değilse klasik DFT uygula (fallback)
    if n & (n - 1) != 0:
        k = np.arange(n)
        matrix = np.exp(-2j * np.pi * np.outer(k, k) / n)
        return np.dot(matrix, x)
        
    if n <= 1:
        return x
        
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])
    
    factor = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + factor[:n // 2] * odd,
                           even + factor[n // 2:] * odd])

def ifft1d(x: np.ndarray) -> np.ndarray:
    """Ters 1D Fourier Dönüşümü (IFFT)."""
    # IFFT bağıntısı: ifft(x) = conj(fft(conj(x))) / N
    return np.conj(fft1d(np.conj(x))) / len(x)

def fft2d(image: np.ndarray) -> np.ndarray:
    """
    2D Fourier Dönüşümü:
    Ayrılabilirlik özelliği sayesinde önce satırlara, sonra sütunlara 1D FFT uygulanır.
    """
    img_complex = image.astype(complex)
    # 1. Her satıra 1D FFT
    row_fft = np.apply_along_axis(fft1d, axis=1, arr=img_complex)
    # 2. Her sütuna 1D FFT
    col_fft = np.apply_along_axis(fft1d, axis=0, arr=row_fft)
    return col_fft

def ifft2d(freq_data: np.ndarray) -> np.ndarray:
    """2D Ters Fourier Dönüşümü (IFFT2)."""
    row_ifft = np.apply_along_axis(ifft1d, axis=1, arr=freq_data)
    col_ifft = np.apply_along_axis(ifft1d, axis=0, arr=row_ifft)
    return np.real(col_ifft)

def fftshift(freq_data: np.ndarray) -> np.ndarray:
    """
    Merkezi matrisin köşelerinden ortaya kaydırır.
    """
    h, w = freq_data.shape
    mid_h, mid_w = h // 2, w // 2
    
    # 4 çeyreği çapraz yer değiştir
    top_left = freq_data[:mid_h, :mid_w]
    top_right = freq_data[:mid_h, mid_w:]
    bottom_left = freq_data[mid_h:, :mid_w]
    bottom_right = freq_data[mid_h:, mid_w:]
    
    top = np.hstack([bottom_right, bottom_left])
    bottom = np.hstack([top_right, top_left])
    return np.vstack([top, bottom])

def ifftshift(freq_data: np.ndarray) -> np.ndarray:
    """fftshift işlemini tersine çevirir (filtrelemeden sonra geri dönmek için)."""
    h, w = freq_data.shape
    mid_h = (h + 1) // 2
    mid_w = (w + 1) // 2
    
    top_left = freq_data[:mid_h, :mid_w]
    top_right = freq_data[:mid_h, mid_w:]
    bottom_left = freq_data[mid_h:, :mid_w]
    bottom_right = freq_data[mid_h:, mid_w:]
    
    top = np.hstack([bottom_right, bottom_left])
    bottom = np.hstack([top_right, top_left])
    return np.vstack([top, bottom])

def apply_frequency_filter(image: np.ndarray, cutoff: float = 30.0, filter_type: str = "low") -> np.ndarray:
    """
    Low-Pass veya High-Pass filtre uygular.
    """
    h, w = image.shape
    center_y, center_x = h // 2, w // 2
    
    # 1. Görüntüyü frekans uzayına taşı ve merkezi ortaya al
    f_transform = fft2d(image)
    f_shifted = fftshift(f_transform)
    
    # 2. Mesafe matrisi ve filtre maskesi oluştur
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    if filter_type == "low":
        mask = (dist <= cutoff).astype(float)
    elif filter_type == "high":
        mask = (dist > cutoff).astype(float)
    else:
        raise ValueError("filter_type 'low' veya 'high' olmalıdır.")
        
    # 3. Maskeyi frekans spektrumuna uygula
    filtered_shifted = f_shifted * mask
    
    # 4. Merkeze geri taşı ve uzamsal düzleme (uzay düzlemine) geri dön (IFFT)
    filtered_f = ifftshift(filtered_shifted)
    filtered_image = ifft2d(filtered_f)
    
    return np.clip(filtered_image, 0, 255)