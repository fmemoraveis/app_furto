import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import seaborn as sns
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans

import scipy
from scipy import stats

import pydeck as pdk
import plotly.express as px


import warnings
warnings.filterwarnings("ignore")

import pickle

# Lendo o dataset de furtos para calculo dos clusters

df_clusters = pd.read_csv('coordenadas_furtos.csv')

#st.write(df_clusters)

# load model
model = pickle.load(open('model.pkl','rb'))

model_coord = pickle.load(open('model_coord.pkl','rb'))

# apresentar números com 0 casas decimais
pd.set_option('display.float_format', lambda x: '%.0f' % x)

@st.cache(allow_output_mutation=True)
def end_to_coord(endereco):

    geolocator = Nominatim()
    location = geolocator.geocode(endereco)

    df_coord = pd.DataFrame({'lat': location.latitude,
                        'lon': location.longitude,
                        'temp' : [0] })

    del df_coord['temp']

    return(df_coord)

@st.cache(allow_output_mutation=True)   
def conf_localizacao(df_coord):
    
    geolocalizacao = df_coord.assign(COORDENADAS = df_coord.lat.astype(str) + ', ' + df_coord.lon.astype(str))
    geolocator = Nominatim(timeout=None)
    location = geolocator.reverse(geolocalizacao[0:1]['COORDENADAS'])
    #location.address
    return(location.address)

# Função que roda o KMeans

@st.cache(allow_output_mutation=True)
def knn(df,df_end):
   

    kmeans = KMeans(n_clusters = 50, init ='k-means++')

    model_knn = kmeans.fit(df[df.columns[0:2]])

    df_clusters['cluster_label'] = kmeans.predict(df[df.columns[0:2]])

    centers = kmeans.cluster_centers_

    #st.write(centers)

    cluster = kmeans.predict(df_end[0:2])

    #st.write(df_end[0:2])

    #st.write(cluster)

    return(df_clusters,cluster)


@st.cache(suppress_st_warning=True)
def des_clusters(df):# ainda não funcionou
    #import shapely.geometry
    #geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
    #geo_df = gpd.GeoDataFrame(df,geometry = geometry)

    st.write(df)

    x1 = df['LONGITUDE']
    x2 = df['LATITUDE']

    fig= plt.figure()

    plt.scatter(x1,x2, c=y,alpha=0.8)

    #geo_df.plot(column='cluster_label',ax=ax,alpha=0.5, legend=True, markersize=15)

    plt.title('Cidade de São Paulo Ocorrência de furtos', fontsize=15,fontweight='bold')

    st.pyplot()

    return()

# Inicio do corpo do sistema

text = """
**  Probabilidade de fruto de veículos**\n
*Município de São Paulo*\n """

st.sidebar.markdown(text)



#Checkbox
st.sidebar.subheader("Localização do Veículo")


lat = ''
lon = ''
peso = 0

rua = st.sidebar.text_input(label="Rua / Avenida")

numero = st.sidebar.text_input(label="Número")

cidade = "São Paulo"

endereco = (numero+' '+rua+' '+cidade)

df_coorde = end_to_coord(endereco)
endereco = conf_localizacao(df_coorde)
#st.sidebar.text(endereco)


if (st.sidebar.checkbox("Mapa de São Paulo",False)):
    df_coorde = end_to_coord(endereco)
    st.text(endereco)
    

    #co-ordinates for the initial view state should be somewhere in Ny ad not the world map
    midpoint = np.average(df_coorde['lat']), np.average(df_coorde['lon'])
    #creating a pydeck figure(an empty 3D map)
    st.write(pdk.Deck(
        map_style = "mapbox://styles/mapbox/light-v9",
        initial_view_state = {
            "latitude":midpoint[0],
            "longitude":midpoint[1],
            "zoom":11,
            "pitch":25#refers to the camera angle of view for a 3D plot
            }, #creating a 3D layer on top of the map/fig
        ))
        
#  Calculando o peso do endereço solicitado

if st.sidebar.button("Mapa e Cluster"):
    df_clusters, cluster = knn(df_clusters,df_coorde)

    #if (st.sidebar.checkbox("Mapa de Clusters",False)):
    import shapely.geometry
    geometry = [Point(xy) for xy in zip(df_clusters['LONGITUDE'], df_clusters['LATITUDE'])]
    geo_df = gpd.GeoDataFrame(df_clusters,geometry = geometry)
    #x1 = df_clusters['LONGITUDE']
    #x2 = df_clusters['LATITUDE']
    fig, ax = plt.subplots()
    geo_df.plot(column='cluster_label',ax=ax,alpha=0.5, legend=True, markersize=15)
    #plt.scatter(x1,x2,alpha=0.8)
    plt.title('Cidade de São Paulo Ocorrência de furtos', fontsize=15,fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    st.pyplot()
    st.text('Cluster do endereco:')
    st.text(cluster)

all_modelos = ['HB20','ONIX','GOL','Saveiro']
modelo = st.sidebar.selectbox("Modelo do veículo",all_modelos)

ano = st.sidebar.text_input("Ano de fabricação do veículo")

dias_semana = ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo']
dia_semana = st.sidebar.selectbox("Dia da Semana",dias_semana)
st.sidebar.text(dia_semana)

horario = st.sidebar.slider( "Horário",0,23)

if(horario < 6):
    periodo = 1
    txt_periodo = 'Madrugada'
elif(horario >= 6 and horario < 12):
    periodo = 2
    txt_periodo = 'Manhã'
elif(horario >= 12 and horario < 18):
    periodo = 3
    txt_periodo = 'Tarde'
elif(horario >= 18):
    periodo = 4
    txt_periodo = 'Noite'

if st.sidebar.button("Submeter"):
    
    st.write('Modelo do Veículo:', modelo)

    st.write('Ano de fabricação:', ano)

    st.write('Localização:',endereco)

    st.write('Dia da Semana:',dia_semana)

    st.write('Horário:',horario)

    st.write('Periodo do dia:',txt_periodo)


    df_prob = pd.DataFrame({'hora_leit': horario,
                            'periodo_leit': periodo,
                            'hora_gprs': horario,
                            'periodo_gprs': periodo, 
                            'peso': peso,
                            'temp' : [0] })

    del df_prob['temp']

    df_prob.head()

    probability_class_1 = model.predict_proba(df_prob)[0, 1]
    st.write('Probabilidade de seu veiculo ser furtado: ',probability_class_1*100, '%')  
	
