import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression, PassiveAggressiveRegressor, ElasticNet, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import validation_curve, LeaveOneOut, train_test_split, cross_val_score
from sklearn.model_selection import cross_validate, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import pickle
import warnings

# Layout for logo
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('https://i.postimg.cc/BQ67tSft/sedapinvest2.png')

with col3:
    st.write(' ')

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set visualization parameters
sns.set(rc={'figure.figsize':(8,8)})
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']

try:
    # Read data
    data_train = pd.read_excel('Rekapitulasi Data (Concise).xlsx')
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    
    # Display debugging information at startup
    if st.checkbox("Show dataset information (debugging)"):
        st.write("Dataset shape:", data_train.shape)
        st.write("Dataset columns:", data_train.columns.tolist())
        st.write("Sample row from dataset:")
        st.write(data_train.iloc[0])
        
        # Display information about provinces and years
        st.write("Available provinces:", data_train['Provinsi'].unique())
        st.write("Available years:", data_train['Tahun'].unique())
        
    # Convert data types
    data_train['PDRB-1'] = data_train['PDRB-1'].astype('int64')
    data_train['J_Penduduk'] = data_train['J_Penduduk'].astype('int64')
    data_train['PriInv'] = data_train['PriInv'].astype('int64')

    # Initialize one-hot encoder
    ohc = OneHotEncoder(handle_unknown='ignore')

    # Fit and transform
    new_features = ohc.fit_transform(data_train[['Provinsi']])

    # Create dataframe with results
    cols_name = [x for x in data_train['Provinsi'].unique()]
    prov = pd.DataFrame(new_features.toarray(), columns=cols_name)

    # Combine with original data
    inv1 = pd.concat([data_train, prov], axis=1)
    data_train1 = inv1.drop(['Provinsi', 'Tahun', 'UMP-1'], axis=1)
    
    # Show model input features if debugging is enabled
    if st.checkbox("Show model features (debugging)"):
        X = data_train1.drop(['PriInv'], axis=1)
        st.write("Model input features:", X.columns.tolist())
    
    # Sidebar logo
    st.sidebar.image("https://i.postimg.cc/wTGfmC92/sedaplogo.png", use_column_width=True)
    menu_utama = st.sidebar.radio('Pilih Menu : ', ('Analisis', 'Prediksi'))

    if menu_utama == 'Analisis':
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://jmp.kemenkeu.go.id/index.php/mapan/article/view/414">Tautan Jurnal</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://s.id/DashboardDDAC">Tautan Dashboard Investasi</a>""", unsafe_allow_html=True)

        # CSS to hide table row index
        hide_table_row_index = """
                    <style>
                    .css-1q8dd3e.edgvbvh1 {display:none}
                    </style>
                    """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

    if menu_utama == 'Prediksi':
        st.markdown("<h1 style='text-align: center; color: #FFCC29; font-family:arial'>Prediksi Jumlah Investasi Agregat (Dalam dan Luar Negeri)</h1>", unsafe_allow_html=True)
        input_pilih_provinsi = st.selectbox('Pilih Provinsi',data_train['Provinsi'].unique())
        
        # Print column names to help debug
        if st.checkbox("Show columns for debugging"):
            st.write("Column names in your dataset:")
            st.write(data_train.columns.tolist())
        
        st.write('#### Data Statis provinsi : ', input_pilih_provinsi)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Infrastruktur</p>", unsafe_allow_html=True)
            
            input_dum_metro = st.number_input('Keberadaan kota metropolitan; 0=tidak ada, 1=ada',key=1,min_value=0, max_value=1,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumMetro'].values[0])
            
            input_dmap = st.number_input('Keberadaan Bandara Utama; 0 = tidak punya 1 =punya',key=2,min_value=0, max_value=1,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dum Main AP'].values[0])
            
            # Input for port quality (try both possible names)
            port_quality_col = 'PortQual' if 'PortQual' in data_train.columns else 'PortQ'
            input_port_q = st.number_input('Kualitas Infrastruktur Pelabuhan', key=3, min_value=0.0, max_value=100.0, 
                                      value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & 
                                                        (data_train['Tahun'] == 2020)][port_quality_col].values[0])
            
            input_infraix = st.number_input('Indeks Komposit Infrastruktur; 0=tak memadai, 1=cukup memadai, 2=sangat memadai',key=4,min_value=0, max_value=2,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Infra Index'].values[0])
            
        with col2:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Geografis</p>", unsafe_allow_html=True)

            input_dist_cp = st.number_input('Jarak Ibukota Provinsi ke Ibukota Negara (dalam km)',step = 1.0,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dist to Cap'].values[0])
            
            input_dist_sg = st.number_input('Jarak Ibukota Provinsi ke Singapura (dalam km)', step = 1.0,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dist to SG'].values[0])
            
            input_dum_oil = st.number_input('Cadangan Minyak Bumi; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumOil'].values[0])
            
            input_dum_ng = st.number_input('Cadangan Gas Alam; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumNG'].values[0])
            
            input_dum_coal = st.number_input('Cadangan Batu Bara; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumCoal'].values[0])
            
        st.markdown("<h4 style='text-align: center; color: #ffffff; font-family:arial'>Data Dinamis Provinsi (Data autofill tahun 2020): </h4>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Belanja Pemerintah</p>", unsafe_allow_html=True)
            input_belanja_51 = st.number_input('Belanja Pegawai', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['51G'].values[0])
            finput_belanja_51 = "Rp{:,d}".format(input_belanja_51)
            st.write(finput_belanja_51)

            input_belanja_52 = st.number_input('Belanja Barang dan Jasa', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['52G'].values[0])
            finput_belanja_52 = "Rp{:,d}".format(input_belanja_52)
            st.write(finput_belanja_52)
            
            input_belanja_53 = st.number_input('Belanja Modal', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['53G'].values[0])
            finput_belanja_53 = "Rp{:,d}".format(input_belanja_53)
            st.write(finput_belanja_53)
            
            input_belanja_lain = st.number_input('Belanja Lainnya Selain Transfer Gabungan Pusat & Daerah', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['OtEG'].values[0])
            finput_belanja_lain = "Rp{:,d}".format(input_belanja_lain)
            st.write(finput_belanja_lain)

        with col2:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Perekonomian</p>", unsafe_allow_html=True)
            
            input_umr_1 = st.number_input('UMR Tahun Sebelumnya', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['UMP-1'].values[0])
            finput_umr_1 = "Rp{:,d}".format(input_umr_1)
            st.write(finput_umr_1)
            
            input_pdrb_1 = st.number_input('PDRB Tahun Sebelumnya', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['PDRB-1'].values[0])
            finput_pdrb_1 = "Rp{:,d}".format(input_pdrb_1)
            st.write(finput_pdrb_1)

        with col3:
            st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Demografis</p>", unsafe_allow_html=True)
            input_j_pen = st.number_input('Jumlah Penduduk', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['J_Penduduk'].values[0])
            finput_j_pen = "{:,d} jiwa".format(input_j_pen)
            st.write(finput_j_pen)

            input_k_pen = st.number_input('Kepadatan Penduduk per km²', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['K_Penduduk'].values[0])
            finput_k_pen = "{:.2f} jiwa per km²".format(input_k_pen)
            st.write(finput_k_pen)

            input_usia_harapan_hidup = st.number_input('Usia Harapan Hidup', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['UHH'].values[0])
            finput_usia_harapan_hidup = "{:.2f} tahun".format(input_usia_harapan_hidup)
            st.write(finput_usia_harapan_hidup)

            # Check which crime column name is used
            crime_cp_col = 'Crime  CP' if 'Crime  CP' in data_train.columns else 'Crime CP'
            input_cri_cp = st.number_input('% Penyelesaian Tindak Pidana', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)][crime_cp_col].values[0])
            finput_cri_cp = "{:.2f}% kasus terselesaikan".format(input_cri_cp)
            st.write(finput_cri_cp)

            input_cri_ri = st.number_input('Risiko Penduduk Terkena Pidana', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Crime Risk'].values[0])
            finput_cri_ri = "{:.2f} per 100.000 penduduk".format(input_cri_ri)
            st.write(finput_cri_ri)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.write(' ')

    with col5:
        if st.button('Prediksi Jumlah Investasi Agregat'):
            # Define X & y
            X = data_train1.drop(['PriInv'], axis=1)
            y = data_train1['PriInv']
            X_columns = X.columns.tolist()
            
            if st.checkbox("Show model feature names (debugging)"):
                st.write("Model feature names:")
                st.write(X_columns)
            
            # Create initial prediction dataframe
            index = [0]
            df_1_pred = pd.DataFrame({
                'provinsi': input_pilih_provinsi,
                'DumMetro': input_dum_metro,
                'Dum Main AP': input_dmap,
                port_quality_col: input_port_q,  # Use the column name from the dataset
                'Infra Index': input_infraix,
                'Dist to Cap': input_dist_cp,
                'Dist to SG': input_dist_sg,
                'DumOil': input_dum_oil,
                'DumNG': input_dum_ng,
                'DumCoal': input_dum_coal,
                '51G': input_belanja_51,
                '52G': input_belanja_52,
                '53G': input_belanja_53,
                'OtEG': input_belanja_lain,
                'UMP-1': input_umr_1,
                'PDRB-1': input_pdrb_1,
                'J_Penduduk': input_j_pen,
                'K_Penduduk': input_k_pen,
                'UHH': input_usia_harapan_hidup,
                crime_cp_col: input_cri_cp,  # Use the column name from the dataset
                'Crime Risk': input_cri_ri
            }, index=index)
            
            # Create prediction data with exactly the same columns as model expects
            df_kosong_1 = pd.DataFrame(0, index=[0], columns=X_columns)
            
            # Define column mappings (input column name to model column name)
            col_map = {
                'DumMetro': 'DumMetro',
                'Dum Main AP': 'Dum Main AP',
                'PortQual': 'PortQual',
                'PortQ': 'PortQ',
                'Infra Index': 'Infra Index',
                'Dist to Cap': 'Dist to Cap',
                'Dist to SG': 'Dist to SG',
                'DumOil': 'DumOil',
                'DumNG': 'DumNG',
                'DumCoal': 'DumCoal',
                '51G': '51G',
                '52G': '52G',
                '53G': '53G',
                'OtEG': 'OtEG',
                'PDRB-1': 'PDRB-1',
                'J_Penduduk': 'J_Penduduk',
                'K_Penduduk': 'K_Penduduk',
                'UHH': 'UHH',
                'Crime CP': 'Crime CP',
                'Crime  CP': 'Crime  CP',
                'Crime Risk': 'Crime Risk'
            }
            
            # First try direct mapping for columns
            for col in df_1_pred.columns:
                if col in X_columns:
                    df_kosong_1[col] = df_1_pred[col]
                elif col in col_map and col_map[col] in X_columns:
                    df_kosong_1[col_map[col]] = df_1_pred[col]
            
            # Handle special cases with potential naming differences
            if 'PortQual' in X_columns and 'PortQ' in df_1_pred.columns:
                df_kosong_1['PortQual'] = df_1_pred['PortQ']
            elif 'PortQ' in X_columns and 'PortQual' in df_1_pred.columns:
                df_kosong_1['PortQ'] = df_1_pred['PortQual']
                
            if 'Crime  CP' in X_columns and 'Crime CP' in df_1_pred.columns:
                df_kosong_1['Crime  CP'] = df_1_pred['Crime CP']
            elif 'Crime CP' in X_columns and 'Crime  CP' in df_1_pred.columns:
                df_kosong_1['Crime CP'] = df_1_pred['Crime  CP']
            
            # Debug: show prediction data before prediction
            if st.checkbox("Show prediction data (debugging)"):
                st.write("First 5 columns of prediction data:")
                st.write(df_kosong_1.iloc[:, :5])
            
            # Check if all feature column names match
            missing_cols = set(X_columns) - set(df_kosong_1.columns)
            extra_cols = set(df_kosong_1.columns) - set(X_columns)
            
            if missing_cols:
                st.error(f"Missing columns in prediction data: {missing_cols}")
            if extra_cols:
                st.warning(f"Extra columns in prediction data (will be ignored): {extra_cols}")
                
            # Make predictions only if all required columns are present
            if not missing_cols:
                try:
                    # Ensure all columns in df_kosong_1 match the order in X_columns
                    df_kosong_1 = df_kosong_1[X_columns]
                    
                    pred_1 = loaded_model.predict(df_kosong_1)
                    investasli = data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['PriInv'].values[0]
                    pred_selisih = pred_1 - investasli
                    
                    st.write('Prediksi investasi berdasar data diatas adalah : ')
                    st.write("## Rp" + f'{pred_1[0]:,}')

                    st.write('Investasi riil berdasar data diatas adalah : ')
                    st.write("## Rp" + f'{investasli:,}')

                    st.write('Selisih Prediksi investasi berdasar data diatas adalah : ')
                    st.write("## Rp" + f'{pred_selisih[0]:,}')
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.write("Model columns:", X_columns)
                    st.write("Prediction columns:", df_kosong_1.columns.tolist())
                    
                    # Show detailed column analysis
                    st.write("### Detailed column analysis:")
                    for col in X_columns:
                        if col in df_kosong_1.columns:
                            st.write(f"✅ {col} - present")
                        else:
                            st.write(f"❌ {col} - missing")
            else:
                st.error("Cannot make prediction due to missing columns")

    with col6:
        st.write(' ')
        
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write("Please make sure all required files are uploaded (Rekapitulasi Data (Concise).xlsx and model.pkl)")
    
    # Show more detailed error information
    import traceback
    st.code(traceback.format_exc())
