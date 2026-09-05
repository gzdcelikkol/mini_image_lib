# Mini Görüntü ve Sinyal İşleme Kütüphanesi

Bu kütüphane; görüntü işleme ve sinyal analizi alanındaki temel algoritmaları harici görüntü işleme kütüphaneleri (OpenCV, SciPy vb.) kullanmadan, **yalnızca saf Python ve NumPy matris operasyonları** ile sıfırdan uygulamak üzere geliştirilmiştir.

---

## 1. 2D Konvolüsyon ve Padding (`core/conv.py`)

Konvolüsyon, uzamsal düzlemde bir filtre çekirdeğinin (kernel) görüntü üzerinde kaydırılarak yerel piksellerin ağırlıklı toplamının alınması işlemidir.

* **Matematiksel İfade:**
  $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) \cdot K(m, n)$$
* **Detay:** Matematiksel konvolüsyon tanımı gereği çekirdek hem yatay hem dikey olarak 180 derece döndürülür (`np.flip`). Kenar kayıplarını ve boyut uyuşmazlığını önlemek adına yansıtma (`reflect`) dolgusu uygulanmıştır.

---

## 2. Kenar Tespiti ve Türev Filtreleri (`core/edges.py`)

* **Gauss Çekirdeği:** Görüntüdeki yüksek frekanslı gürültüleri bastırmak için kullanılır.
  $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
* **Sobel Filtresi:** Görüntünün $x$ ve $y$ eksenlerindeki yönlü türevlerini (gradyanlarını) ayrık fark matrisleriyle hesaplar:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
* **Gradyan Büyüklüğü ve Yönü:**
  $$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan2(G_y, G_x)$$
* **Basitleştirilmiş Canny:** Gauss yumuşatmasının ardından Sobel gradyanı hesaplanır ve çift eşikleme (double thresholding) ile güçlü ve zayıf kenarlar sınıflandırılır.

---

## 3. Morfolojik İşlemler (`core/morphology.py`)

Morfolojik operatörler, yapılandırıcı bir eleman ($B$) altındaki küme işlemlerine dayanır.

* **Erosion (Aşındırma):** Yerel minimumu alır; nesne sınırlarını inceltir ve izole beyaz pikselleri yok eder:
  $$\operatorname{Erosion}(I, B)(x, y) = \min_{(i, j) \in B} I(x + i, y + j)$$
* **Dilation (Genişletme):** Yerel maksimumu alır; nesne sınırlarını genişletir ve iç delikleri kapatır:
  $$\operatorname{Dilation}(I, B)(x, y) = \max_{(i, j) \in B} I(x - i, y - j)$$
* **Opening (Açma):** $\operatorname{Dilate}(\operatorname{Erode}(I))$ — Küçük dış parazitleri yok eder.
* **Closing (Kapama):** $\operatorname{Erode}(\operatorname{Dilate}(I))$ — Nesne içindeki siyah çatlakları ve delikleri kapatır.

---

## 4. Fourier Dönüşümü ve Frekans Filtreleri (`core/fourier.py`)

* **1D Ayrık Fourier Dönüşümü (DFT) & Cooley-Tukey FFT:**  
  Zaman/uzay düzlemindeki sinyali frekans bileşenlerine ayrıştırır:
  $$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}$$
  Cooley-Tukey Radix-2 algoritması sinyali tek ve çift indekslere bölerek karmaşıklığı $O(N^2)$ seviyesinden $O(N \log N)$ seviyesine düşürür.
* **2D Fourier Dönüşümü:**  
  Ayrılabilirlik özelliği sayesinde önce her satıra 1D FFT, çıkan sonuca da sütun bazında 1D FFT uygulanır:
  $$F(u, v) = \mathcal{F}_{\text{col}}\left(\mathcal{F}_{\text{row}}(I)\right)$$
* **Frekans Düzleminde Filtreleme:**  
  Frekans spektrumu merkeze taşındıktan (`fftshift`) sonra merkeze olan Öklid mesafesine ($D(u, v)$) göre maskeleme yapılır:
  * **Low-Pass (Alçak Geçiren):** $D(u, v) \le D_0$ geçirilir (görüntüyü yumuşatır/bulanıklaştırır).
  * **High-Pass (Yüksek Geçiren):** $D(u, v) > D_0$ geçirilir (kenarları ve ani değişimleri yakalar).
  Ters FFT (`ifft2d`) ile görüntü tekrar uzamsal düzleme dönüştürülür.

---

## Çalıştırma

Gereksinimleri yükleyin ve testi başlatın:

```bash
pip install -r requirements.txt
python main.py
