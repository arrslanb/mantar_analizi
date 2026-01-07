# mantar_analizi
#  Mantar Analiz ve Tespit Sistemi (Mushroom Classification AI)

Bu proje, makine öğrenmesi algoritmaları kullanılarak mantarların zehirli olup olmadığını tespit eden bir yapay zeka uygulamasıdır.

##  Proje Hakkında
* Veri Seti: UCI Mushroom Dataset (8.124 Örnek)
* Kullanılan Modeller: Random Forest (%100 Başarı), Decision Tree (%100 Başarı), Logistic Regression.
* Arayüz: Gradio (Web Tabanlı)
* Dil:Python

## Sonuçlar
Projemizde **Random Forest** algoritması seçilmiş ve test verileri üzerinde **%100 doğruluk** oranına ulaşılmıştır.

## Kurulum ve Çalıştırma
1. Bu repoyu indirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas numpy scikit-learn gradio matplotlib seaborn
