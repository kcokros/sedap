import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import catboost
from sklearn.linear_model import HuberRegressor, LinearRegression, PassiveAggressiveRegressor, ElasticNet, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.model_selection import validation_curve, LeaveOneOut, train_test_split, cross_val_score
from sklearn.model_selection import cross_validate, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import pyplot
import pickle

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('https://i.postimg.cc/BQ67tSft/sedapinvest2.png')

with col3:
    st.write(' ')


# #Baca Dataset awal
data_train = pd.read_excel('./Rekapitulasi Data (Concise).xlsx')
loaded_model = pickle.load(open('./model.sav', 'rb'))



#Disable Warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#Set Size
sns.set(rc={'figure.figsize':(8,8)})
#Coloring
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']

#st.markdown("<h1 style='text-align: center; color: #243A74; font-family:sans-serif'>Job Market Analysis for Post Covid-19 Economic Recovery</h1>", unsafe_allow_html=True)

st.sidebar.image: st.sidebar.image("https://i.postimg.cc/wTGfmC92/sedaplogo.png", use_column_width=True)
menu_utama = st.sidebar.radio('Pilih Menu : ', ('Analisis', 'Prediksi'))

if menu_utama == 'Analisis':
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://ddac2022.com/docpaper/50.%20PAPER_SEDAP%20-%20Muhamad%20Ameer%20Noor.pdf">Tautan Jurnal</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center; font-size:40px; ' href="https://s.id/DashboardDDAC">Tautan Dashboard Investasi</a>""", unsafe_allow_html=True)

        def main(): html_temp = """<div class='tableauPlaceholder' id='viz1647010468490' style='position: relative'><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DashboardDDAC&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div> <script type='text/javascript'> var divElement = document.getElementById('viz1647010468490'); var vizElement = divElement.getElementsByTagName('object')[0]; if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='1927px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='1927px';} else { vizElement.style.width='100%';vizElement.style.height='5577px';} var scriptElement = document.createElement('script'); scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js'; vizElement.parentNode.insertBefore(scriptElement, vizElement);</script>"""


if menu_utama == 'Prediksi':
                st.markdown("<h1 style='text-align: center; color: #FFCC29; font-family:arial'>Prediksi Jumlah Investasi Agregat (Dalam dan Luar Negeri)</h1>", unsafe_allow_html=True)
                input_pilih_provinsi = st.selectbox('Pilih Provinsi',data_train['Provinsi'].unique())
                for item1 in data_train['Provinsi'].unique():
                    if item1 == input_pilih_provinsi:
                        st.write(input_pilih_provinsi)
                input_pilih_tahun = st.selectbox('Pilih Provinsi',data_train['Tahun'].unique())
                for item2 in data_train['Tahun'].unique():
                    if item2 == input_pilih_tahun:
                        st.write(input_pilih_tahun)

                st.write('#### Berikut adalah data yang terisi secara otomatis untuk provinsi : ', input_pilih_provinsi)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Infrastruktur</p>", unsafe_allow_html=True)
                    
                    input_dum_metro = st.selectbox('Keberadaan kota metropolitan; 0=tidak ada, 1=ada',data_train['DumMetro'].unique())
                    for item3 in data_train['DumMetro'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_dum_metro)

                    input_dmap = st.selectbox('Keberadaan Bandara Utama; 0 = tidak punya 1 =punya',data_train['Dum Main AP'].unique())
                    for item4 in data_train['Provinsi'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_dmap)

                    input_port_q = st.number_input('Kualitas Infrastruktur Pelabuhan', min_value=0.0, max_value=1.0, value=0.316, step=0.01)
                    for item5 in data_train['PortQ'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_port_q)

                    input_infraix = st.selectbox('Indeks Komposit Infrastruktur; 0=tidak memadai, 1=cukup memadai, 2=sangat memadai',data_train['Infra Index'].unique())
                    for item6 in data_train['Infra Index'].unique():
                        if item1 == input_pilih_provinsi and data_train['Tahun'] == 2020:
                            st.write(input_infraix)

                with col2:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Geografis</p>", unsafe_allow_html=True)

                    input_dist_cp = st.number_input('Jarak Ibukota Provinsi ke Ibukota Negara (dalam km)',step = 1.0,value= data_train[(data_train['Provinsi'] == input_pilih_provinsi) & (data_train['Tahun'] == 2020)]['Dist to Cap'].values[0])
                    for item6 in data_train['Dist to Cap'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dist_cp)

                    input_dist_sg = st.number_input('Jarak Ibukota Provinsi ke Singapura (dalam km)', min_value=0, max_value=10000, value=1830, step=1)
                    for item7 in data_train['Provinsi'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dist_sg)

                    input_dum_oil = st.selectbox('Cadangan Minyak Bumi; 0=sedikit, 1=banyak',data_train['DumOil'].unique())
                    for item8 in data_train['Provinsi'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_oil)

                    input_dum_ng = st.selectbox('Cadangan Gas Alam; 0=sedikit, 1=banyak',data_train['DumNG'].unique())
                    for item9 in data_train['Provinsi'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_ng)

                    input_dum_coal = st.selectbox('Cadangan Batu Bara; 0=sedikit, 1=banyak',data_train['DumCoal'].unique())
                    for item10 in data_train['Provinsi'].unique():
                        if item1 == input_pilih_provinsi and data_train.Tahun == 2020:
                            st.write(input_dum_coal)

                st.markdown("<h4 style='text-align: center; color: #ffffff; font-family:arial'>Lengkapi data berikut: </h4>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Belanja Pemerintah</p>", unsafe_allow_html=True)
                    input_belanja_51 = st.number_input('Belanja Pegawai', min_value=1000000, max_value=100000000000000, value=19285486099020, step=1000)
                    input_belanja_52 = st.number_input('Belanja Barang dan Jasa', min_value=1000000, max_value=100000000000000, value=13258660858311, step=1000)
                    input_belanja_53 = st.number_input('Belanja Modal', min_value=1000000, max_value=100000000000000, value=8492556614925, step=1000)
                    input_belanja_lain = st.number_input('Belanja Lainnya Selain Transfer Gabungan Pusat & Daerah', min_value=1000000, max_value=100000000000000, value=3428386300303, step=1000)
                with col2:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Perekonomian</p>", unsafe_allow_html=True)
                    input_umr_1 = st.number_input('Upah Minimum Regional Tahun Sebelumnya', min_value=1000, max_value=100000000000000, value=2916810, step=1000)
                    input_pdrb_1 = st.number_input('Produk Domestik Regional Bruto Tahun Sebelumnya', min_value=1000, value=164162980000000, step=1000)
                
                with col3:
                    st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi Demografis</p>", unsafe_allow_html=True)
                    input_j_pen = st.number_input('Jumlah Penduduk', min_value=1000, max_value=100000000, value=5388100, step=100)
                    input_k_pen = st.number_input('Kepadatan Penduduk per Kilometer Persegi', min_value=1, max_value=100000000, value=97, step=1)
                    input_usia_harapan_hidup = st.number_input('Usia Harapan Hidup', min_value=20, max_value=120, value=60, step=1)
                    input_cri_cp = st.number_input('Persentase Penyelesaian Tindak Pidana*', min_value=0.0, max_value=10000.0, value=37.9, step=0.1)
                    input_cri_ri = st.number_input('Risiko Penduduk Terkena Tindak Pidana (Per 100.000 Penduduk)', min_value=1, max_value=1000000, value=149, step=1)




                if st.button('Prediksi Jumlah Investasi Agregat'):
                    #Split Kolom provinsi
                    # inisiasi encoder 
                    ohc = OneHotEncoder(handle_unknown='ignore')
                    # fit dan transform
                    new_features = ohc.fit_transform(data_train[['provinsi']])
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in data_train['provinsi'].unique()]
                    loc = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([data_train,loc], axis=1)

                    ##Split Kolom Tahun
                    # inisiasi encoder 
                    #ohc = OneHotEncoder(handle_unknown='ignore')
                    ## fit dan transform
                    #new_features = ohc.fit_transform(data_train[['provinsi']])
                    ## masukan ke dataframe hasilnya
                    #cols_name= [x for x in data_train['provinsi'].unique()]
                    #loc = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    ## gabung hasilnya ke invest_clean
                    #invest_clean = pd.concat([data_train,loc], axis=1)

                    #Split Kolom DumMetro
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['DumMetro'].unique()]
                    dmetro = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,dmetro], axis=1)

                    #Split Kolom 51G
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['51G'].unique()]
                    belanja_51 = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,belanja_51], axis=1)

                    #Split Kolom category
                    # fit dan transform
                    new_features = ohc.fit_transform(invest_clean[['category']])
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['category'].unique()]
                    cat = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,cat], axis=1)

                    #Split Kolom company_industry
                    # fit dan transform
                    new_features = ohc.fit_transform(invest_clean[['company_industry']])
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['company_industry'].unique()]
                    company_industry = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,company_industry], axis=1)
                    
                    #Split Kolom company_size
                    # fit dan transform
                    new_features = ohc.fit_transform(invest_clean[['company_size']])
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['company_size'].unique()]
                    company_size = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,company_size], axis=1)
                    #Split Kolom pend_min
                    # fit dan transform
                    new_features = ohc.fit_transform(invest_clean[['pend_min']])
                    # masukan ke dataframe hasilnya
                    cols_name= [x for x in invest_clean['pend_min'].unique()]
                    pend_min = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,pend_min], axis=1)
                    #Split Kolom Tahun
                    # fit dan transform
                        ##new_features = ohc.fit_transform(invest_clean[['Tahun']])
                    # masukan ke dataframe hasilnya
                        ##cols_name= [x for x in invest_clean['Tahun'].unique()]
                        ##Tahun = pd.DataFrame(new_features.toarray(), columns=cols_name)
                    # gabung hasilnya ke invest_clean
                    invest_clean = pd.concat([invest_clean,Tahun], axis=1)
                    invest_clean_ok = invest_clean.drop(['Provinsi','Tahun','category','company_industry','company_size','pend_min','Tahun'],axis=1)
                    #define X & y
                    X = invest_clean_ok.drop(['salary_ave'], axis=1)
                    y = invest_clean_ok['salary_ave']
                    index=[0]
                    df_1_pred = pd.DataFrame({
                        'provinsi' : input_pilih_provinsi,
                        'Tahun' : input_pilih_tahun,
                        'DumMetro' : input_dum_metro,
                        'Dum Main AP' : input_dmap,
                        'PortQ' : input_port_q,
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
                # model_PIet = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                #     max_depth=27, max_features='auto', max_leaf_nodes=None,
                #     max_samples=None, min_impurity_decrease=0.0,
                #     min_impurity_split=None, min_samples_leaf=1,
                #     min_samples_split=2, min_weight_fraction_leaf=0.0,
                #     n_estimators=25, n_jobs=-1, oob_score=False,
                #     random_state=7744, verbose=0, warm_start=False)
                # model_PIet.fit(X,y)
                pred_1 = loaded_model.predict(df_kosong_1)
                #investasi = 
                st.write('Prediksi investasi berdasar data diatas adalah : ', pred_1)

