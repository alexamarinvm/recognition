#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sklearn 
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import urllib.request
#from IPython.display import display, Image
from sklearn.metrics import pairwise_distances
from datetime import datetime
import streamlit as st


# In[2]:


# Para revisar directorios
import os

print(os.getcwd())


# In[3]:


# Directorios (Cambiar por el propio cuando se descarguen las bases)
path = "C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE"
 # Directorio general donde se depositaran las caracteristicas e indices de las imagenes
try:
   os.chdir(path)
   print("Directorio actual: {0}".format(os.getcwd()))
except FileNotFoundError:
   print("Directorio: {0} no existe".format(path))
except NotADirectoryError:
   print("{0} no es un directorio".format(path))
except PermissionError:
   print("Sin permiso de modificar {0}".format(path))


# In[4]:


st.set_option('deprecation.showfileUploaderEncoding', False)

# Uso de las características extraídas por categoría

houses_df = pd.read_excel("C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE/houses.xlsx")
contemporary_features = np.load('C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE/CONTEMPORARY_ResNet_features.npy')
mediterranean_features = np.load('C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE/MEDITERRANEAN_ResNet_features.npy')
traditional_features = np.load('C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE/TRADITIONAL_ResNet_features.npy')
ranch_features = np.load('C:/Users/alexa/OneDrive/CONCENTRACION_LIT/RETOFASE2/COMPLETO_DESPLIEGUE/RANCH_ResNet_features.npy')
houses_df["HouseId"] = houses_df["HouseId"].astype(str)

# Indices de los objetos por categoría


houses_contemporary = houses_df[houses_df["Style"]=="Contemporary/Modern"]
houses_ranch = houses_df[houses_df["Style"]=="Ranch"]
houses_traditional = houses_df[houses_df["Style"]=="Traditional"]
houses_mediterranean = houses_df[houses_df["Style"]=="Mediterranean"]



# In[5]:


def get_similar_products_cnn(product_id, num_results): # Primero definir la categoria
    if(houses_df[houses_df['HouseId']==product_id]['Style'].values[0]=="Contemporary/Modern"):
        extracted_features = contemporary_features
        Productids = list(houses_contemporary['HouseId'])
    elif(houses_df[houses_df['HouseId']==product_id]['Style'].values[0]=="Ranch"):
        extracted_features = ranch_features
        Productids = list(houses_ranch['HouseId'])
    elif(houses_df[houses_df['HouseId']==product_id]['Style'].values[0]=="Mediterranean"):
        extracted_features = mediterranean_features
        Productids = list(houses_mediterranean['HouseId'])
    elif(houses_df[houses_df['HouseId']==product_id]['Style'].values[0]=="Traditional"):
        extracted_features = traditional_features
        Productids = list(houses_traditional['HouseId'])
    Productids = list(Productids)
    doc_id = Productids.index(product_id) # Id de los productos
    pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    st.write("""
         #### Details 
         """)
    ip_row = houses_df[['ImageURL','HouseTitle']].loc[houses_df['HouseId']==Productids[indices[0]]]
    for indx, row in ip_row.iterrows():
        image = Image.open(urllib.request.urlopen(row['ImageURL']))
        image = image.resize((224,224))
        st.image(image)
        st.write(f"Chosen House : {row['HouseTitle']}")
    st.write(f"""
         #### The {num_results} houses recommended for you are: 
         """)
    for i in range(1,len(indices)):
        rows = houses_df[['ImageURL','HouseTitle']].loc[houses_df['HouseId']==Productids[indices[i]]]
        for indx, row in rows.iterrows():
            #image = Image.open(Image(url=row['ImageURL'], width = 224, height = 224,embed=True))
            image = Image.open(urllib.request.urlopen(row['ImageURL']))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"House Title: {row['HouseTitle']}")
            st.write(f"Distance of the image: {pdists[i]}")

st.write("""
         ## House Recommendation
         """
         )


user_input1 = st.text_input("Insert the Id of the house you liked the most")
user_input2 = st.text_input("Insert the number of recommendations you would like to see")


button = st.button('Generate recommendations')
if button:
    get_similar_products_cnn(str(user_input1), int(user_input2))


# Para la ejecucion del codigo, escribir en la terminar de VisualStudio: streamlit run SistemaRecomendacion_Despliegue.py
# Se desplegara un recuadro en una ventana de su navegador, ingresaran el id del producto junto con la cantidad de 
# muestras que se desean ver y se mostrarán los resultados de la ejecución de la red

# %%
