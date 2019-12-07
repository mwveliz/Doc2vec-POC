# -*- coding: latin-1 -*-
"""
Elaborado por:
    Miguel Delgado
    Paula Aguirre
    Fabian Palma
    
Licencia GPLv3 
"""

        
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from gensim import models

def iniciar_datos():
    #Leyendo e csv a un data frame
    df = pd.read_csv('df_EmpleoPublico.csv', encoding="latin1")
    
    #Conviertiendo los nan a vacio para evitar problemas de parseo
    df = df.replace(np.nan, '', regex=True)
    
    #Mostrando los nombres de las columnas
    names = df.columns.values.tolist()
    print(names)
    
    #Mostrando los primeros registros
    data = df.values
    print(data)
    
    #Combiando las dos columnas del dataframe en 1 Cargo + Objetivo del Cargo -> busqueda
    df['busqueda'] = df[['Cargo', 'Objetivo del Cargo']].apply(lambda x: ' '.join(x), axis=1)
    #Transformando a minuscula la columna de busqueda para facilitar la busqueda de palabras
    df['busqueda'] = df['busqueda'].str.lower() 
    
    #Elimnando  las stopwords en español
    final_stopwords_list = stopwords.words('spanish') 
    df['busqueda'] = df['busqueda'].apply(lambda x:' '.join(x for x in x.split() if not x in final_stopwords_list))
    
    #Eliminando los siignos de puntuacion
    df['busqueda'] = df['busqueda'].apply(lambda x:' '.join(x for x in x.split()  if x not in string.punctuation))
    
    #Obteniendo los cargos unicos
    cargos_unicos = df['Cargo'].unique().tolist()
    
    len(cargos_unicos) # cargos unicos
    len( df['Cargo'])-len(cargos_unicos) # cargos repetidos
    
    
    return df


def iniciar_modelo(df):
    
    busqueda = df['busqueda'].tolist()
    cargo = df['Cargo'].tolist()
    
    docs = []
    #Iterando sobre la lista de busqueda etiqueto con el cargo para retornarlo
    for i in range(len(busqueda)):
        sent = models.doc2vec.LabeledSentence(words = busqueda[i].split(),tags = [cargo[i]])
        docs.append(sent) #agregando a la lista de docs para modelar
        
    #Configurando el modelo Doc2Vec    
    model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1) 
    
    #construyndo el vocabulario con la lista de docs etiquetados
    model.build_vocab(docs)
    #Mejorando el performance
    model.init_sims(replace=True)
    return model


def buscar(df, model, querysearch):
    querysearch = querysearch.lower()
    #obteniendo los tags (cargos)
        
    try:
        modelsearch = model[querysearch]
    except:
        print('Lo sentimos, no hay resultados, intente con algo mas amplio')
        return 0
    resultados = model.docvecs.most_similar(positive=[modelsearch])
    
    for resultado in resultados:
        salida = []
        if resultado[1]>=UMBRAL:
            salida =  df[df['Cargo'] == resultado[0]]
            for index, row in salida.iterrows():
                print('******************** Oportunidad***************')
                print('Cargo: ', row['Cargo'])
                print('Ciudad: ', row['Ciudad'])
                print('Salario: ', row['WAGE'])        
                #print('Institucion: ', row['Cargo'])
                print('Ministerio: ', row['Ministerio'])
                #print('No de Vacantes: ', row['Cargo'])        
                #print('Región: ', row['Cargo'])        
                print('***********************************************')        
                print('\n') 
    return 1
            
        

#Definiendo umbral de similitud para reportar resultados
UMBRAL = 0.34

#Iniciando el dataframe
print('Iniciando dataframe...')
df = iniciar_datos()

#Inicinado el modelo
print('Iniciando modelo...')

model = iniciar_modelo(df)
print('Bienvenido al motor avanzado de busqueda de empleo')
#print("Antes de comenzar seteemos un umbral de busqueda, el recomendado es 0.34")
#u = input("ingrese un umbral o presione enter para dejar el actual")
        
        
while True:
    q = input("Ingrese su palabra o palabras de busqueda de empleo:  (escriba exit para salir) y presione enter \n ")
    if q == 'exit':
        break
    t = buscar(df, model, q)
print("Gracias por venir, vuelva pronto")