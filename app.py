import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Dashboard Status Desa", layout="wide")

@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('model_status_desa.sav')
        scaler = joblib.load('scaler_desa.sav')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat file model (.sav): {e}")
        return None, None

@st.cache_data
def load_data():
    files = {
        "sekolah": "jrk_sekolah.csv",
        "ekonomi": "keberadaan_simpan_pinjam.csv",
        "olahraga": "lapangan_olahraga.csv",
        "sinyal": "sinyal_telepon.csv",
        "perikanan": "teknologi_perikanan.csv",
        "peternakan": "teknologi_peternakan.csv",
        "sampah": "tempat_sampah.csv", 
        "nakes": "tenaga_kesehatan.csv"
    }

    for f in files.values():
        if not os.path.exists(f):
            st.error(f"File {f} tidak ditemukan di repositori GitHub!")
            return None

    try:
        dfs = {name: pd.read_csv(path) for name, path in files.items()}
        
        key_cols = ['bps_kode_desa_kelurahan', 'tahun']
        
        df = dfs['sekolah'][['bps_kode_desa_kelurahan', 'bps_nama_kabupaten_kota', 'bps_nama_desa_kelurahan', 'tahun', 'jarak_sd_terdekat']]
        
        merge_map = {
            'ekonomi': 'keberadaan_bumdesa_keuangan_program_usaha_ekonomi_desa',
            'olahraga': 'jumlah_fasilitas_olahraga',
            'sinyal': 'status_sinyal_telepon_seluler',
            'perikanan': 'jumlah_alat_teknologi_tepat_guna_perikanan',
            'peternakan': 'jumlah_alat_teknologi_tepat_guna_peternakan',
            'sampah': 'ketersediaan_tempat_pembuangan_sampah',
            'nakes': 'jumlah_tenaga_kesehatan_lainnya'
        }
        
        for name, col in merge_map.items():
            subset = dfs[name][key_cols + [col]]
            df = pd.merge(df, subset, on=key_cols, how='inner')
        
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Gagal menggabungkan data: {e}")
        return None

st.title("📊 Dashboard Klastering Status Desa")
st.markdown("Aplikasi analisis otomatis berbasis Machine Learning untuk klasifikasi status desa.")

model, scaler = load_ml_components()
df_raw = load_data()

if model and df_raw is not None:
    sinyal_map = {'SINYAL KUAT': 3, 'SINYAL LEMAH': 2, 'TIDAK ADA SINYAL': 1}
    df_raw['sinyal_score'] = df_raw['status_sinyal_telepon_seluler'].map(sinyal_map).fillna(0)
    df_raw['bumdes_score'] = df_raw['keberadaan_bumdesa_keuangan_program_usaha_ekonomi_desa'].apply(lambda x: 1 if x == 'ADA' else 0)
    df_raw['sampah_score'] = df_raw['ketersediaan_tempat_pembuangan_sampah'].apply(lambda x: 1 if x == 'ADA' else 0)
    
    features = ['jarak_sd_terdekat', 'bumdes_score', 'jumlah_fasilitas_olahraga', 'sinyal_score', 
                'jumlah_alat_teknologi_tepat_guna_perikanan', 'jumlah_alat_teknologi_tepat_guna_peternakan', 
                'sampah_score', 'jumlah_tenaga_kesehatan_lainnya']
    
    try:
        X_scaled = scaler.transform(df_raw[features])
        df_raw['cluster'] = model.predict(X_scaled)
        
        status_map = {0: "Berkembang", 1: "Maju", 2: "Tertinggal", 3: "Sangat Tertinggal", 4: "Mandiri"}
        df_raw['status_desa'] = df_raw['cluster'].map(status_map)

        st.success(f"Berhasil menganalisis {len(df_raw)} desa.")
        
        tab1, tab2, tab3 = st.tabs(["📈 Ringkasan Klaster", "🗺️ Sebaran Wilayah", "📋 Data Detail"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(df_raw, names='status_desa', title="Persentase Status Desa",
                                 color='status_desa', color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                dist_table = df_raw['status_desa'].value_counts().reset_index()
                dist_table.columns = ['Status', 'Jumlah']
                st.write("Tabel Distribusi")
                st.dataframe(dist_table, use_container_width=True)

        with tab2:
            st.subheader("Sebaran Status Desa per Kabupaten")
            fig_bar = px.histogram(df_raw, x="bps_nama_kabupaten_kota", color="status_desa", 
                                   barmode="stack", title="Distribusi Wilayah",
                                   height=600)
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.dataframe(df_raw[['bps_nama_kabupaten_kota', 'bps_nama_desa_kelurahan', 'status_desa'] + features])

    except Exception as e:
        st.error(f"Gagal melakukan prediksi klaster: {e}")