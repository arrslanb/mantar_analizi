import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

# --- 1. VERİ YÜKLEME ---
try:
    data = pd.read_csv('mushrooms.csv')
    print("✅ Veri başarıyla yüklendi.")
except FileNotFoundError:
    print("❌ HATA: mushrooms.csv dosyası bulunamadı! Dosyayı kodun yanına koy.")
    exit()

original_data = data.copy()

# Raporlama için veriyi sayıya çevir
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

X = data.drop(['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. HOCANIN İSTEDİĞİ KARŞILAŞTIRMA ---
print("⏳ Modeller eğitiliyor (Karşılaştırmalı Analiz)...")
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ {name} Başarısı: %{results[name]*100:.2f}")

# --- 3. TÜRKÇE - İNGİLİZCE SÖZLÜKLERİ ---
sekil_tr = {
    "🔔 Çan Şeklinde (Bell)": "b", "🔼 Konik (Conical)": "c",
    "⚫ Dışbükey (Convex)": "x", "➖ Düz (Flat)": "f",
    "👊 Yumrulu (Knobbed)": "k", "🔽 Çökük (Sunken)": "s"
}
koku_tr = {
    "👃 Badem (Almond)": "a", "🌿 Anason (Anise)": "l",
    "🔥 Yanık (Creosote)": "c", "🐟 Balık Gibi (Fishy)": "y",
    "🤢 İğrenç (Foul)": "f", "🍄 Küflü (Musty)": "m",
    "😶 Kokusuz (None)": "n", "🌶️ Keskin (Pungent)": "p",
    "🥘 Baharatlı (Spicy)": "s"
}
renk_tr = {
    "🟤 Kahverengi": "n", "🟡 Sarımsı": "b", "🧱 Tarçın": "c", "⚪ Gri": "g",
    "🟢 Yeşil": "r", "🌸 Pembe": "p", "🟣 Mor": "u", "🔴 Kırmızı": "e",
    "☁️ Beyaz": "w", "☀️ Sarı": "y"
}

# --- 4. ARAYÜZ MODELİ (Random Forest) ---
demo_cols = ['cap-shape', 'odor', 'cap-color']
X_demo = original_data[demo_cols].copy()
y_demo = original_data['class']
encoders = {}
for col in demo_cols:
    le_demo = LabelEncoder()
    X_demo[col] = le_demo.fit_transform(X_demo[col]) 
    encoders[col] = le_demo

le_target = LabelEncoder()
y_demo_enc = le_target.fit_transform(y_demo)
demo_model = RandomForestClassifier()
demo_model.fit(X_demo, y_demo_enc)

# --- 5. TAHMİN FONKSİYONU ---
def mantar_analiz(sekil_secim, koku_secim, renk_secim):
    try:
        # Seçim yapılmadıysa uyarı
        if not sekil_secim or not koku_secim or not renk_secim:
            return "⚠️ Lütfen tüm kutucukları seçiniz."

        # Seçilen Türkçeyi harf koduna, sonra sayıya çevir
        val_shape = encoders['cap-shape'].transform([sekil_tr[sekil_secim]])[0]
        val_odor = encoders['odor'].transform([koku_tr[koku_secim]])[0]
        val_color = encoders['cap-color'].transform([renk_tr[renk_secim]])[0]
        
        # Tahmin et
        tahmin = demo_model.predict([[val_shape, val_odor, val_color]])[0]
        sonuc = le_target.inverse_transform([tahmin])[0]
        
        if sonuc == 'p':
            return "☠️ SAKIN YEME ZEHİRLİ"
        else:
            return "✅ YENEBİLİR GÜVENLİ"
            
    except Exception as e:
        return f"Beklenmedik bir hata oluştu: {str(e)}"

# --- 6. ARAYÜZ (SAFE MODE) ---
# Buradaki title ve description kısmı raporda görünür, yeterlidir.
with gr.Blocks(title="Mantar Analiz Sistemi") as interface:
    gr.Markdown("# 🍄 Mantar Analiz ve Tahmin Sistemi")
    gr.Markdown("**Proje:** Mantarın Şekil, Koku ve Renk özelliklerine göre zehirli olup olmadığını tespit eden Yapay Zeka uygulaması.")
    
    with gr.Row():
        with gr.Column():
            inp_sekil = gr.Dropdown(choices=list(sekil_tr.keys()), label="1. Şapka Şekli")
            inp_koku = gr.Dropdown(choices=list(koku_tr.keys()), label="2. Koku")
            inp_renk = gr.Dropdown(choices=list(renk_tr.keys()), label="3. Renk")
            btn = gr.Button("🔍 ANALİZ ET", variant="primary")
        
        with gr.Column():
            out_text = gr.Textbox(label="📊 Analiz Sonucu", lines=2)

    btn.click(fn=mantar_analiz, inputs=[inp_sekil, inp_koku, inp_renk], outputs=out_text)

print("\nUygulama başlatılıyor... Linke tıkla!")
interface.launch(share=True)