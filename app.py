import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sistem Analisis Desa", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model_status_desa.sav')
    scaler = joblib.load('scaler_desa.sav')
    df = pd.read_csv('dataset_desa_final.csv')
    return model, scaler, df

try:
    model_fixed, scaler_fixed, df_master = load_assets()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard Utama (Fixed k=5)", "🧪 Lab Dinamis (Custom k)"])

feature_order = [
    'jarak_sekolah_terdekat', 'bumdes_score', 'jumlah_fasilitas_olahraga', 
    'sinyal_score', 'jumlah_alat_teknologi_tepat_guna_perikanan', 
    'jumlah_alat_teknologi_tepat_guna_peternakan', 'sampah_score', 
    'jumlah_tenaga_kesehatan_lainnya'
]

if menu == "🏠 Dashboard Utama (Fixed k=5)":
    st.title("🏛️ Dashboard Analisis Status Desa")
    kab_list = ["Semua Kabupaten"] + sorted(df_master['bps_nama_kabupaten_kota'].unique().tolist())
    selected_kab = st.sidebar.selectbox("Pilih Wilayah:", kab_list)
    df_filtered = df_master if selected_kab == "Semua Kabupaten" else df_master[df_master['bps_nama_kabupaten_kota'] == selected_kab]

    st.divider()
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("📌 Distribusi Status")
        dist = df_filtered['status_desa'].value_counts().reset_index()
        st.dataframe(dist, use_container_width=True)
    with col2:
        fig_pie = px.pie(dist, values='count', names='status_desa', hole=0.4,
                         color='status_desa',
                         color_discrete_map={
                             'Mandiri': '#1b5e20', 'Maju': '#4caf50', 
                             'Berkembang': '#2196f3', 'Tertinggal': '#ff9800', 'Sangat Tertinggal': '#f44336'
                         })
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🗺️ Sebaran Status Per Kabupaten")
    fig_bar = px.histogram(df_filtered, x="bps_nama_kabupaten_kota", color="status_desa", barmode="stack",
                           color_discrete_map={
                             'Mandiri': '#1b5e20', 'Maju': '#4caf50', 
                             'Berkembang': '#2196f3', 'Tertinggal': '#ff9800', 'Sangat Tertinggal': '#f44336'
                           })
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.title("🧪 Laboratorium Eksperimen Klastering")
    st.info("Atur jumlah klaster melalui slider untuk melihat perubahan distribusi Centroid secara dinamis.")
    
    k_custom = st.slider("Pilih Jumlah Klaster (k):", min_value=2, max_value=10, value=5)
    
    df_lab = df_master.copy()
    num_cols = ['jarak_sekolah_terdekat', 'jumlah_fasilitas_olahraga', 
                'jumlah_alat_teknologi_tepat_guna_perikanan', 
                'jumlah_alat_teknologi_tepat_guna_peternakan', 
                'jumlah_tenaga_kesehatan_lainnya']
    
    df_log = df_lab.copy()
    for col in num_cols:
        df_log[col] = np.log1p(df_log[col])
    
    X_scaled = scaler_fixed.transform(df_log[feature_order])
    
    kmeans_new = KMeans(n_clusters=k_custom, init='k-means++', random_state=42, n_init=10)
    df_lab['new_cluster'] = kmeans_new.fit_predict(X_scaled)
    df_lab['new_label'] = df_lab['new_cluster'].apply(lambda x: f"Cluster {x}")

    st.divider()

    st.subheader(f"🎯 Visualisasi Sebaran & Centroid (k={k_custom})")
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    pca_centroids = pca.transform(kmeans_new.cluster_centers_)
    
    fig_pca = go.Figure()

    fig_pca.add_trace(go.Scatter(
        x=pca_data[:, 0], y=pca_data[:, 1],
        mode='markers',
        marker=dict(color=df_lab['new_cluster'], colorscale='Viridis', size=6, opacity=0.5),
        text=df_lab['bps_nama_desa_kelurahan'],
        name='Data Desa'
    ))

    fig_pca.add_trace(go.Scatter(
        x=pca_centroids[:, 0], y=pca_centroids[:, 1],
        mode='markers',
        marker=dict(color='red', size=14, symbol='x', line=dict(width=2, color='white')),
        name='Titik Pusat (Centroid)'
    ))

    fig_pca.update_layout(
        xaxis_title="Principal Component 1", yaxis_title="Principal Component 2",
        height=600, showlegend=True,
        template="plotly_white"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()

    st.subheader(f"🗺️ Sebaran Wilayah Berdasarkan k={k_custom}")
    fig_bar_lab = px.histogram(df_lab, x="bps_nama_kabupaten_kota", color="new_label", barmode="stack",
                               color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_bar_lab, use_container_width=True)

    st.subheader("📋 Karakteristik Rata-rata per Klaster")
    profile = df_lab.groupby('new_label')[feature_order].mean().reset_index()
    st.dataframe(profile.style.highlight_max(axis=0, color='#d4edda').highlight_min(axis=0, color='#f8d7da'), use_container_width=True)

st.divider()
with st.expander("📋 Lihat Data Detail"):
    st.dataframe(df_master if menu == "🏠 Dashboard Utama (Fixed k=5)" else df_lab)

