import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Kemandirian Desa Jabar",
    page_icon="🏘️",
    layout="wide"
)

# --- FUNGSI LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('hasil_clustering_desa_fixed.csv')
    return df

@st.cache_resource
def train_model(df):
    features = ['skor_simpan_pinjam', 'skor_sinyal', 
                'jumlah_alat_teknologi_tepat_guna_perikanan', 
                'jumlah_alat_teknologi_tepat_guna_peternakan', 
                'skor_kesehatan']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means dengan 3 Cluster
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)
    
    return model, scaler, features

# Load Data
try:
    df = load_data()
    model, scaler, feature_cols = train_model(df)
    
    # Mapping Nama Cluster
    cluster_names = {
        0: "Desa Digital Konsumtif",
        1: "Desa Maju Sejahtera",
        2: "Desa Pedalaman Produktif"
    }
    # Fallback jika cluster id berbeda saat training ulang
    df['label_cluster'] = df['cluster'].map(cluster_names).fillna("Cluster Lain")
    
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.stop()

# --- SIDEBAR (INPUT & FILTER) ---
st.sidebar.title("🎛️ Simulasi Desa Baru")
input_simpan = st.sidebar.selectbox("Ada Simpan Pinjam (BUMDes)?", ["TIDAK ADA", "ADA"])
input_sinyal = st.sidebar.selectbox("Kualitas Sinyal", ["TIDAK ADA SINYAL", "SINYAL LEMAH", "SINYAL KUAT"])
input_ikan = st.sidebar.number_input("Jml Alat Perikanan", 0, 1000, 5)
input_ternak = st.sidebar.number_input("Jml Alat Peternakan", 0, 1000, 5)
input_sehat = st.sidebar.slider("Skor Akses Kesehatan (Kabupaten)", 1.0, 5.0, 4.0)

# Konversi input
val_simpan = 1 if input_simpan == "ADA" else 0
val_sinyal = 3 if input_sinyal == "SINYAL KUAT" else (2 if input_sinyal == "SINYAL LEMAH" else 1)

if st.sidebar.button("Prediksi Cluster"):
    input_data = [[val_simpan, val_sinyal, input_ikan, input_ternak, input_sehat]]
    input_scaled = scaler.transform(input_data)
    pred_cluster = model.predict(input_scaled)[0]
    pred_label = cluster_names.get(pred_cluster, f"Cluster {pred_cluster}")
    st.sidebar.success(f"Hasil: **{pred_label}**")

# --- HALAMAN UTAMA ---
st.title("🏘️ Dashboard Analisis Klasterisasi Desa")

# 1. METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Total Desa", f"{len(df):,}")
col2.metric("Rata-rata Alat Ikan", f"{df['jumlah_alat_teknologi_tepat_guna_perikanan'].mean():.1f}")
col3.metric("Rata-rata Skor Sehat", f"{df['skor_kesehatan'].mean():.2f}")

st.divider()

# 2. VISUALISASI UTAMA
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🗺️ Sebaran Karakteristik")
    # FIX: Menghapus hover_data yang memanggil nama desa (karena tidak ada di CSV)
    fig_scatter = px.scatter(
        df, 
        x="jumlah_alat_teknologi_tepat_guna_perikanan", 
        y="skor_kesehatan",
        color="label_cluster",
        size="skor_sinyal",
        # Kita ganti hover dengan data fitur saja agar aman
        hover_data=['jumlah_alat_teknologi_tepat_guna_peternakan', 'skor_simpan_pinjam'],
        title="Hubungan Alat Produksi vs Akses Kesehatan",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 Proporsi Cluster")
    cluster_counts = df['label_cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Jumlah']
    fig_pie = px.pie(cluster_counts, values='Jumlah', names='Cluster', hole=0.4)
    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# 3. DATA TABLE
st.subheader("📂 Data Mentah (Preview)")
# FIX: Tidak menampilkan kolom Nama, hanya Kode dan Fitur
cols_to_show = ['kemendagri_kode_desa_kelurahan', 'label_cluster'] + feature_cols
# Cek apakah kolom ada sebelum menampilkan
valid_cols = [c for c in cols_to_show if c in df.columns]
st.dataframe(df[valid_cols])