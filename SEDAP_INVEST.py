import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import pickle
import warnings

# Title and logo
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image('https://i.postimg.cc/BQ67tSft/sedapinvest2.png')
with col3:
    st.write(' ')

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(rc={'figure.figsize':(8,8)})

try:
    # Read data
    data_train = pd.read_excel('Rekapitulasi Data (Concise).xlsx')
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    
    # Convert data types
    data_train['PDRB-1'] = data_train['PDRB-1'].astype('int64')
    data_train['J_Penduduk'] = data_train['J_Penduduk'].astype('int64')
    data_train['PriInv'] = data_train['PriInv'].astype('int64')

    # One-hot encoding
    ohc = OneHotEncoder(handle_unknown='ignore')
    new_features = ohc.fit_transform(data_train[['Provinsi']])
    cols_name = [x for x in data_train['Provinsi'].unique()]
    prov = pd.DataFrame(new_features.toarray(), columns=cols_name)
    inv1 = pd.concat([data_train, prov], axis=1)
    data_train1 = inv1.drop(['Provinsi', 'Tahun', 'UMP-1'], axis=1)
    
    # Sidebar
    st.sidebar.image("https://i.postimg.cc/wTGfmC92/sedaplogo.png", use_column_width=True)
    menu_utama = st.sidebar.radio('Pilih Menu : ', ('Analisis', 'Prediksi'))

    if menu_utama == 'Analisis':
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://jmp.kemenkeu.go.id/index.php/mapan/article/view/414">Tautan Jurnal</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://s.id/DashboardDDAC">Tautan Dashboard Investasi</a>""", unsafe_allow_html=True)

        hide_table_row_index = """
                    <style>
                    .css-1q8dd3e.edgvbvh1 {display:none}
                    </style>
                    """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

    if menu_utama == 'Prediksi':
        st.markdown("<h1 style='text-align: center; color: #FFCC29; font-family:arial'>Prediksi Jumlah Investasi Agregat (Dalam dan Luar Negeri)</h1>", unsafe_allow_html=True)
        
        # Debug information
        if st.checkbox("Show Debug Information"):
            st.write("Model columns (first 5):")
            X_test = data_train1.drop(['PriInv'], axis=1)
            st.write(X_test.columns.tolist()[:5])
        
        # Province selection
        input_pilih_provinsi = st.selectbox('Pilih Provinsi', data_train['Provinsi'].unique())
        
        st.write('#### Data Statis provinsi : ', input_pilih_provinsi)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Infrastruktur</p>", unsafe_allow_html=True)
            
            input_dum_metro = st.number_input('Keberadaan kota metropolitan; 0=tidak ada, 1=ada', 
                                             key=1, min_value=0, max_value=1,
                                             value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                              (data_train['Tahun'] == 2020)]['DumMetro'].values[0])
            
            input_dmap = st.number_input('Keberadaan Bandara Utama; 0 = tidak punya 1 =punya',
                                        key=2, min_value=0, max_value=1,
                                        value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                         (data_train['Tahun'] == 2020)]['Dum Main AP'].values[0])
            
            input_port_q = st.number_input('Kualitas Infrastruktur Pelabuhan', 
                                          key=3, min_value=0.0, max_value=100.0,
                                          value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                           (data_train['Tahun'] == 2020)]['PortQual'].values[0])
            
            input_infraix = st.number_input('Indeks Komposit Infrastruktur; 0=tak memadai, 1=cukup memadai, 2=sangat memadai',
                                           key=4, min_value=0, max_value=2,
                                           value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                            (data_train['Tahun'] == 2020)]['Infra Index'].values[0])
            
        with col2:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Geografis</p>", unsafe_allow_html=True)

            input_dist_cp = st.number_input('Jarak Ibukota Provinsi ke Ibukota Negara (dalam km)',
                                           step=1.0,
                                           value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                            (data_train['Tahun'] == 2020)]['Dist to Cap'].values[0])
            
            input_dist_sg = st.number_input('Jarak Ibukota Provinsi ke Singapura (dalam km)', 
                                           step=1.0,
                                           value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                            (data_train['Tahun'] == 2020)]['Dist to SG'].values[0])
            
            input_dum_oil = st.number_input('Cadangan Minyak Bumi; 0=sedikit, 1=banyak', 
                                           min_value=0, max_value=1,
                                           value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                            (data_train['Tahun'] == 2020)]['DumOil'].values[0])
            
            input_dum_ng = st.number_input('Cadangan Gas Alam; 0=sedikit, 1=banyak', 
                                          min_value=0, max_value=1,
                                          value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                           (data_train['Tahun'] == 2020)]['DumNG'].values[0])
            
            input_dum_coal = st.number_input('Cadangan Batu Bara; 0=sedikit, 1=banyak', 
                                            min_value=0, max_value=1,
                                            value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                             (data_train['Tahun'] == 2020)]['DumCoal'].values[0])
            
        st.markdown("<h4 style='text-align: center; color: #ffffff; font-family:arial'>Data Dinamis Provinsi (Data autofill tahun 2020): </h4>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Belanja Pemerintah</p>", unsafe_allow_html=True)
            input_belanja_51 = st.number_input('Belanja Pegawai', 
                                              value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                               (data_train['Tahun'] == 2020)]['51G'].values[0])
            finput_belanja_51 = "Rp{:,d}".format(input_belanja_51)
            st.write(finput_belanja_51)

            input_belanja_52 = st.number_input('Belanja Barang dan Jasa', 
                                              value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                               (data_train['Tahun'] == 2020)]['52G'].values[0])
            finput_belanja_52 = "Rp{:,d}".format(input_belanja_52)
            st.write(finput_belanja_52)
            
            input_belanja_53 = st.number_input('Belanja Modal', 
                                              value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                               (data_train['Tahun'] == 2020)]['53G'].values[0])
            finput_belanja_53 = "Rp{:,d}".format(input_belanja_53)
            st.write(finput_belanja_53)
            
            input_belanja_lain = st.number_input('Belanja Lainnya Selain Transfer Gabungan Pusat & Daerah', 
                                                value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                                 (data_train['Tahun'] == 2020)]['OtEG'].values[0])
            finput_belanja_lain = "Rp{:,d}".format(input_belanja_lain)
            st.write(finput_belanja_lain)

        with col2:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Perekonomian</p>", unsafe_allow_html=True)
            
            input_umr_1 = st.number_input('UMR Tahun Sebelumnya', 
                                         value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                          (data_train['Tahun'] == 2020)]['UMP-1'].values[0])
            finput_umr_1 = "Rp{:,d}".format(input_umr_1)
            st.write(finput_umr_1)
            
            input_pdrb_1 = st.number_input('PDRB Tahun Sebelumnya', 
                                          value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                           (data_train['Tahun'] == 2020)]['PDRB-1'].values[0])
            finput_pdrb_1 = "Rp{:,d}".format(input_pdrb_1)
            st.write(finput_pdrb_1)

        with col3:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Demografis</p>", unsafe_allow_html=True)
            input_j_pen = st.number_input('Jumlah Penduduk', 
                                         value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                          (data_train['Tahun'] == 2020)]['J_Penduduk'].values[0])
            finput_j_pen = "{:,d} jiwa".format(input_j_pen)
            st.write(finput_j_pen)

            input_k_pen = st.number_input('Kepadatan Penduduk per km²', 
                                         value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                          (data_train['Tahun'] == 2020)]['K_Penduduk'].values[0])
            finput_k_pen = "{:.2f} jiwa per km²".format(input_k_pen)
            st.write(finput_k_pen)

            input_usia_harapan_hidup = st.number_input('Usia Harapan Hidup', 
                                                      value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                                       (data_train['Tahun'] == 2020)]['UHH'].values[0])
            finput_usia_harapan_hidup = "{:.2f} tahun".format(input_usia_harapan_hidup)
            st.write(finput_usia_harapan_hidup)

            input_cri_cp = st.number_input('% Penyelesaian Tindak Pidana', 
                                          value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                           (data_train['Tahun'] == 2020)]['Crime  CP'].values[0])
            finput_cri_cp = "{:.2f}% kasus terselesaikan".format(input_cri_cp)
            st.write(finput_cri_cp)

            input_cri_ri = st.number_input('Risiko Penduduk Terkena Pidana', 
                                          value=data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                           (data_train['Tahun'] == 2020)]['Crime Risk'].values[0])
            finput_cri_ri = "{:.2f} per 100.000 penduduk".format(input_cri_ri)
            st.write(finput_cri_ri)

        col4, col5, col6 = st.columns(3)

        with col4:
            st.write(' ')

        with col5:
            if st.button('Prediksi Jumlah Investasi Agregat'):
                # Get model features
                X = data_train1.drop(['PriInv'], axis=1)
                y = data_train1['PriInv']
                
                # Important: Get the exact column order from the model
                model_columns = X.columns.tolist()
                
                # Create prediction dataframe with the exact same column structure as training data
                # First create a template from the first row of X
                df_kosong_1 = X[:1].copy()
                
                # Initialize with zeros
                for col in df_kosong_1.columns:
                    df_kosong_1[col].values[:] = 0
                
                # Fill in values from user inputs
                # For categorical variables (provinces)
                if input_pilih_provinsi in df_kosong_1.columns:
                    df_kosong_1[input_pilih_provinsi] = 1
                
                # Fill in the remaining values
                df_kosong_1['DumMetro'] = input_dum_metro
                df_kosong_1['Dum Main AP'] = input_dmap
                df_kosong_1['PortQual'] = input_port_q
                df_kosong_1['Infra Index'] = input_infraix
                df_kosong_1['Dist to Cap'] = input_dist_cp
                df_kosong_1['Dist to SG'] = input_dist_sg
                df_kosong_1['DumOil'] = input_dum_oil
                df_kosong_1['DumNG'] = input_dum_ng
                df_kosong_1['DumCoal'] = input_dum_coal
                df_kosong_1['51G'] = input_belanja_51
                df_kosong_1['52G'] = input_belanja_52
                df_kosong_1['53G'] = input_belanja_53
                df_kosong_1['OtEG'] = input_belanja_lain
                df_kosong_1['PDRB-1'] = input_pdrb_1
                df_kosong_1['J_Penduduk'] = input_j_pen
                df_kosong_1['K_Penduduk'] = input_k_pen
                df_kosong_1['UHH'] = input_usia_harapan_hidup
                df_kosong_1['Crime  CP'] = input_cri_cp
                df_kosong_1['Crime Risk'] = input_cri_ri
                
                # Debug: Show what's being sent to the model
                if st.checkbox("Show prediction data"):
                    st.write("Prediction data:")
                    st.write(df_kosong_1)
                
                try:
                    # Make prediction
                    pred_1 = loaded_model.predict(df_kosong_1)
                    investasli = data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                           (data_train['Tahun'] == 2020)]['PriInv'].values[0]
                    pred_selisih = pred_1 - investasli
                    
                    st.write('Prediksi investasi berdasar data diatas adalah : ')
                    st.write("## Rp{:,}".format(int(pred_1[0])))
                    
                    st.write('Investasi riil berdasar data diatas adalah : ')
                    st.write("## Rp{:,}".format(investasli))
                    
                    st.write('Selisih Prediksi investasi berdasar data diatas adalah : ')
                    st.write("## Rp{:,}".format(int(pred_selisih[0])))
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    
                    # Print detailed error information
                    st.write("Model expected columns:", X.columns.tolist())
                    st.write("Columns in prediction data:", df_kosong_1.columns.tolist())
                    
                    # Check for differences
                    missing = set(X.columns) - set(df_kosong_1.columns)
                    extra = set(df_kosong_1.columns) - set(X.columns)
                    if missing:
                        st.error(f"Missing columns: {missing}")
                    if extra:
                        st.warning(f"Extra columns: {extra}")

        with col6:
            st.write(' ')

except Exception as e:
    st.error(f"An error occurred: {e}")
    import traceback
    st.code(traceback.format_exc())
