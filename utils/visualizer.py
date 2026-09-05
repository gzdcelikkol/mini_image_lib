import matplotlib.pyplot as plt
import numpy as np

def plot_before_after(original: np.ndarray, processed: np.ndarray, 
                      title_before: str = "Orijinal", 
                      title_after: str = "Processed", 
                      cmap: str = "gray"):
    """
    Orijinal ve işlenmiş görüntüyü yan yana göstermek için
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Sol taraf: Orijinal görüntü
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title(title_before, fontsize=12)
    axes[0].axis("off")
    
    # Sağ taraf: İşlenmiş görüntü
    axes[1].imshow(processed, cmap=cmap)
    axes[1].set_title(title_after, fontsize=12)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_multiple(images: list, titles: list, cmap: str = "gray"):
    """
    Eğer elimizde 2den fazla görüntü varsa bunları aynı satırda 
    gösterebilmek için gerekli yardımcı fonksiyon.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    if n == 1:
        axes = [axes]
        
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()
    