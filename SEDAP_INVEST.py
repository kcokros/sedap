import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression, PassiveAggressiveRegressor, ElasticNet, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.model_selection import validation_curve, LeaveOneOut, train_test_split, cross_val_score
from sklearn.model_selection import cross_validate, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder 
from matplotlib import pyplot
import pickle
import warnings
import tableauserverclient as TSC
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('https://i.postimg.cc/BQ67tSft/sedapinvest2.png')

with col3:
    st.write(' ')


# #Baca Dataset awal
data_train = pd.read_excel('Rekapitulasi Data (Concise).xlsx')
loaded_model = pickle.load(open('model.pkl', 'rb'))
# load library 

#declare data data float sebagai integer
data_train['PDRB-1'] = data_train['PDRB-1'].astype('int64')
data_train['J_Penduduk'] = data_train['J_Penduduk'].astype('int64')
data_train['PriInv'] = data_train['PriInv'].astype('int64')

# inisiasi encoder 
ohc = OneHotEncoder(handle_unknown='ignore')

# fit dan transform
new_features = ohc.fit_transform(data_train[['Provinsi']])

# masukan ke dataframe hasilnya
cols_name= [x for x in data_train['Provinsi'].unique()]
prov = pd.DataFrame(new_features.toarray(), columns=cols_name)

# gabung hasilnya ke job_clean
inv1 = pd.concat([data_train,prov], axis=1)
data_train1 = inv1.drop(['Provinsi','Tahun','UMP-1'],axis=1)
#Disable Warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#Set Size
sns.set(rc={'figure.figsize':(8,8)})
#Coloring
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']


st.sidebar.image: st.sidebar.image("https://i.postimg.cc/wTGfmC92/sedaplogo.png", use_column_width=True)
menu_utama = st.sidebar.radio('Pilih Menu : ', ('Analisis', 'Prediksi'))

if menu_utama == 'Analisis':
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://jmp.kemenkeu.go.id/index.php/mapan/article/view/414">Tautan Jurnal</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://s.id/DashboardDDAC">Tautan Dashboard Investasi</a>""", unsafe_allow_html=True)

      # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    .css-1q8dd3e.edgvbvh1 {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        

if menu_utama == 'Prediksi':
                st.markdown("<h1 style='text-align: center; color: #FFCC29; font-family:arial'>Prediksi Jumlah Investasi Agregat (Dalam dan Luar Negeri)</h1>", unsafe_allow_html=True)
                input_pilih_provinsi = st.selectbox('Pilih Provinsi',data_train['Provinsi'].unique())
                for item1 in data_train['Provinsi'].unique():
                    if item1 == input_pilih_provinsi:
                        st.write(' ')
                #input_pilih_tahun = st.selectbox('Pilih Provinsi',data_train['Tahun'].unique())
                #for item2 in data_train['Tahun'].unique():
                    #if item2 == input_pilih_tahun:
                        #st.write(input_pilih_tahun)

                st.write('#### Data Statis provinsi : ', input_pilih_provinsi)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Infrastruktur</p>", unsafe_allow_html=True)
                    
                    input_dum_metro = st.number_input('Keberadaan kota metropolitan; 0=tidak ada, 1=ada',key=1,min_value=0, max_value=1,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumMetro'].values[0])
                    for item3 in data_train['DumMetro'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_dum_metro)

                    input_dmap = st.number_input('Keberadaan Bandara Utama; 0 = tidak punya 1 =punya',key=2,min_value=0, max_value=1,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dum Main AP'].values[0])
                    for item4 in data_train['Dum Main AP'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_dmap)

                    input_port_q = st.number_input('Kualitas Infrastruktur Pelabuhan', key=3, min_value=0.0, max_value=100.0, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['PortQual'].values[0])
                    for item5 in data_train['PortQual'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_port_q)

                    input_infraix = st.number_input('Indeks Komposit Infrastruktur; 0=tak memadai, 1=cukup memadai, 2=sangat memadai',key=4,min_value=0, max_value=2,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Infra Index'].values[0])
                    for item6 in data_train['Infra Index'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_infraix)

                with col2:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Geografis</p>", unsafe_allow_html=True)

                    input_dist_cp = st.number_input('Jarak Ibukota Provinsi ke Ibukota Negara (dalam km)',step = 1.0,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dist to Cap'].values[0])
                    for item6 in data_train['Dist to Cap'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dist_cp)

                    input_dist_sg = st.number_input('Jarak Ibukota Provinsi ke Singapura (dalam km)', step = 1.0,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dist to SG'].values[0])
                    for item7 in data_train['Dist to SG'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dist_sg)

                    input_dum_oil = st.number_input('Cadangan Minyak Bumi; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumOil'].values[0])
                    for item8 in data_train['DumOil'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_oil)

                    input_dum_ng = st.number_input('Cadangan Gas Alam; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumNG'].values[0])
                    for item9 in data_train['DumNG'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_ng)

                    input_dum_coal = st.number_input('Cadangan Batu Bara; 0=sedikit, 1=banyak', min_value=0, max_value=1, value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['DumCoal'].values[0])
                    for item10 in data_train['DumCoal'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_coal)

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

                    input_cri_cp = st.number_input('% Penyelesaian Tindak Pidana', value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Crime  CP'].values[0])
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
        #define X & y
        X = data_train1.drop(['PriInv'], axis=1)
        y = data_train1['PriInv']
        index=[0]
        df_1_pred = pd.DataFrame({
            'provinsi' : input_pilih_provinsi,
            'DumMetro' : input_dum_metro,
            'Dum Main AP' : input_dmap,
            'PortQual' : input_port_q,
            'Infra Index' : input_infraix,
            'Dist to Cap' : input_dist_cp,
            'Dist to SG' : input_dist_sg,
            'DumOil' : input_dum_oil,
            'DumNG' : input_dum_ng,
            'DumCoal' : input_dum_coal,
            '51G' : input_belanja_51,
            '52G' : input_belanja_52,
            '53G' : input_belanja_53,
            'OtEG' : input_belanja_lain,
            'UMP-1' : input_umr_1,
            'PDRB-1' : input_pdrb_1,
            'J_Penduduk' : input_j_pen,
            'K_Penduduk' : input_k_pen,
            'UHH' : input_usia_harapan_hidup,
            'Crime CP' : input_cri_cp,
            'Crime Risk' : input_cri_ri
        },index=index)
        #Set semua nilai jadi 0
        df_kosong_1 = X[:1]
        for col in df_kosong_1.columns:
            df_kosong_1[col].values[:] = 0
        list_1 = []
        for i in df_1_pred.columns:
            x = df_1_pred[i][0]
            list_1.append(x)
        #buat dataset baru
        for i in df_kosong_1.columns:
            for j in list_1:
                if i == j:
                    df_kosong_1[i] = df_kosong_1[i].replace(df_kosong_1[i].values,1)  
        df_kosong_1['Crime  CP'] = df_1_pred['Crime CP']   
        df_kosong_1['DumMetro'] = df_1_pred['DumMetro']
        df_kosong_1['Dum Main AP'] = df_1_pred['Dum Main AP']
        df_kosong_1['PortQual'] = df_1_pred['PortQual']
        df_kosong_1['Infra Index'] = df_1_pred['Infra Index']
        df_kosong_1['Dist to Cap' ] = df_1_pred['Dist to Cap' ]
        df_kosong_1['Dist to SG'] = df_1_pred['Dist to SG']
        df_kosong_1['DumOil'] = df_1_pred['DumOil']
        df_kosong_1['DumNG'] = df_1_pred['DumNG']
        df_kosong_1['DumCoal' ] = df_1_pred['DumCoal' ]
        df_kosong_1['51G'] = df_1_pred['51G']
        df_kosong_1['52G']  = df_1_pred['52G' ]
        df_kosong_1['53G'] = df_1_pred['53G']
        df_kosong_1['OtEG' ] = df_1_pred['OtEG' ]
        df_kosong_1['PDRB-1'] = df_1_pred['PDRB-1']
        df_kosong_1['J_Penduduk'] = df_1_pred['J_Penduduk']
        df_kosong_1['K_Penduduk'] = df_1_pred['K_Penduduk']
        df_kosong_1['UHH' ] = df_1_pred['UHH' ]
        df_kosong_1['Crime Risk'] = df_1_pred['Crime Risk']
        pred_1 = loaded_model.predict(df_kosong_1)
        investasli = data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['PriInv'].values[0]
        pred_selisih = pred_1 - investasli
    #investasi = 
        st.write('Prediksi investasi berdasar data diatas adalah : ')
        st.write("## Rp"f'{pred_1[0]:,}')

        st.write('Investasi riil berdasar data diatas adalah : ')
        st.write("## Rp"f'{investasli:,}')

        st.write('Selisih Prediksi investasi berdasar data diatas adalah : ')
        st.write("## Rp"f'{pred_selisih[0]:,}')

with col6:
    st.write(' ')
