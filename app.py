import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Konfigurasi Halaman & Tema
st.set_page_config(page_title="Sistem Analisis Desa Jawa Barat", layout="wide")

# Skema warna standar untuk status desa
COLOR_MAP = {
    'Mandiri': '#1b5e20',          # Hijau Gelap
    'Maju': '#4caf50',             # Hijau
    'Berkembang': '#2196f3',       # Biru
    'Tertinggal': '#ff9800',       # Oranye
    'Sangat Tertinggal': '#f44336' # Merah
}

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_status_desa.sav')
        scaler = joblib.load('scaler_desa.sav')
        df = pd.read_csv('dataset_desa_final.csv')
        return model, scaler, df
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file: {e}. Pastikan file .sav dan .csv ada di folder yang sama.")
        st.stop()

model_fixed, scaler_fixed, df_master = load_assets()

# 2. Sidebar Navigasi
st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard Utama (Fixed k=5)", "🧪 Lab Dinamis (Custom k)"])

# Urutan fitur sesuai saat training model
feature_order = [
    'jarak_sd_terdekat', 'bumdes_score', 'jumlah_fasilitas_olahraga', 
    'sinyal_score', 'jumlah_alat_teknologi_tepat_guna_perikanan', 
    'jumlah_alat_teknologi_tepat_guna_peternakan', 'sampah_score', 
    'jumlah_tenaga_kesehatan_lainnya'
]

# --- HALAMAN 1: DASHBOARD UTAMA ---
if menu == "🏠 Dashboard Utama (Fixed k=5)":
    st.title("🏛️ Dashboard Analisis Status Desa")
    st.markdown("Visualisasi berdasarkan model K-Means yang telah dilatih dengan 5 kategori status desa.")
    
    kab_list = ["Semua Kabupaten"] + sorted(df_master['bps_nama_kabupaten_kota'].unique().tolist())
    selected_kab = st.sidebar.selectbox("Pilih Wilayah:", kab_list)
    
    df_filtered = df_master if selected_kab == "Semua Kabupaten" else df_master[df_master['bps_nama_kabupaten_kota'] == selected_kab]

    st.divider()
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📌 Distribusi Status")
        dist = df_filtered['status_desa'].value_counts().reset_index()
        dist.columns = ['Status Desa', 'Jumlah']
        st.dataframe(dist, use_container_width=True, hide_index=True)
        
    with col2:
        fig_pie = px.pie(dist, values='Jumlah', names='Status Desa', hole=0.4,
                         color='Status Desa', color_discrete_map=COLOR_MAP)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🗺️ Sebaran Status Per Kabupaten")
    fig_bar = px.histogram(df_filtered, x="bps_nama_kabupaten_kota", color="status_desa", 
                           barmode="stack", color_discrete_map=COLOR_MAP,
                           category_orders={"status_desa": ["Mandiri", "Maju", "Berkembang", "Tertinggal", "Sangat Tertinggal"]})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- HALAMAN 2: LAB DINAMIS ---
else:
    st.title("🧪 Laboratorium Eksperimen Klastering")
    st.info("Atur jumlah klaster (k) di sidebar. Sistem akan otomatis melabeli klaster berdasarkan kualitas infrastruktur.")
    
    k_custom = st.sidebar.slider("Pilih Jumlah Klaster (k):", min_value=2, max_value=10, value=5)
    
    df_lab = df_master.copy()
    
    # Preprocessing (Log Transform + Scaling)
    num_cols = ['jarak_sd_terdekat', 'jumlah_fasilitas_olahraga', 
                'jumlah_alat_teknologi_tepat_guna_perikanan', 
                'jumlah_alat_teknologi_tepat_guna_peternakan', 
                'jumlah_tenaga_kesehatan_lainnya']
    
    df_log = df_lab.copy()
    for col in num_cols:
        df_log[col] = np.log1p(df_log[col])
    
    X_scaled = scaler_fixed.transform(df_log[feature_order])
    
    # Menjalankan K-Means baru
    kmeans_new = KMeans(n_clusters=k_custom, init='k-means++', random_state=42, n_init=10)
    df_lab['new_cluster'] = kmeans_new.fit_predict(X_scaled)

    # --- LOGIKA PEMETAAN LABEL DESKRIPTIF ---
    # 1. Hitung skor rata-rata per klaster (semakin tinggi fiturnya, semakin bagus desanya)
    # Khusus 'jarak_sd_terdekat', biasanya semakin kecil semakin bagus, namun kita gunakan rata-rata fitur lainnya sebagai penentu utama
    stats = df_lab.groupby('new_cluster')[feature_order].mean().mean(axis=1)
    rank = stats.sort_values(ascending=False).index # Klaster terbaik di atas
    
    # 2. Siapkan label
    base_labels = ["Mandiri", "Maju", "Berkembang", "Tertinggal", "Sangat Tertinggal"]
    if k_custom > 5:
        base_labels += [f"Cluster Extra {i+1}" for i in range(k_custom - 5)]
    
    # 3. Buat dictionary mapping
    mapping = {cluster_id: base_labels[i] for i, cluster_id in enumerate(rank)}
    df_lab['new_label'] = df_lab['new_cluster'].map(mapping)

    st.divider()

    # Visualisasi PCA
    st.subheader(f"🎯 Visualisasi PCA & Centroid (k={k_custom})")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    pca_centroids = pca.transform(kmeans_new.cluster_centers_)
    
    fig_pca = go.Figure()
    # Titik data desa
    fig_pca.add_trace(go.Scatter(
        x=pca_data[:, 0], y=pca_data[:, 1], mode='markers',
        marker=dict(color=df_lab['new_cluster'], colorscale='Viridis', size=6, opacity=0.6),
        text=df_lab['bps_nama_desa_kelurahan'] + " (" + df_lab['new_label'] + ")",
        name='Data Desa'
    ))
    # Titik Centroid
    fig_pca.add_trace(go.Scatter(
        x=pca_centroids[:, 0], y=pca_centroids[:, 1], mode='markers',
        marker=dict(color='red', size=12, symbol='x', line=dict(width=2, color='white')),
        name='Centroid'
    ))
    fig_pca.update_layout(xaxis_title="PC1", yaxis_title="PC2", template="plotly_white", height=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    # Histogram Sebaran Wilayah Baru
    st.subheader(f"🗺️ Sebaran Wilayah Berdasarkan k={k_custom}")
    fig_bar_lab = px.histogram(df_lab, x="bps_nama_kabupaten_kota", color="new_label", barmode="stack",
                               color_discrete_map=COLOR_MAP,
                               category_orders={"new_label": base_labels})
    st.plotly_chart(fig_bar_lab, use_container_width=True)

    # Tabel Karakteristik
    st.subheader("📋 Karakteristik Rata-rata per Klaster")
    profile = df_lab.groupby('new_label')[feature_order].mean().reindex(
        [v for v in base_labels if v in df_lab['new_label'].unique()]
    )
    st.dataframe(profile.style.highlight_max(axis=0, color='#d4edda').highlight_min(axis=0, color='#f8d7da'), use_container_width=True)

# Footer Detail Data
st.divider()
with st.expander("📋 Lihat Data Detail"):
    st.dataframe(df_master if menu == "🏠 Dashboard Utama (Fixed k=5)" else df_lab)
