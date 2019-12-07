# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
        
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from gensim import models

#Definiendo umbral de similitud para reportar resultados
UMBRAL = 0.34

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

querysearch = 'Santiago'
querysearch = querysearch.lower()
#model.most_similar(querysearch)
#obteniendo los tags (cargos)
resultados = model.docvecs.most_similar(positive=[model[querysearch]])

for resultado in resultados:
    salida = []
    if resultado[1]>=UMBRAL:
        salida =  df[df['Cargo'] == resultado[0]]
        for index, row in salida.iterrows():
            print('******************** Oportunidad***************')
            print('Cargo: ', row['Cargo'])
            print('Ciudad: ', row['Ciudad'])
            print('Salario: ', row['WAGE'])        
            print('***********************************************')        
            print('\n')        
        

for word, row in model.vocab.items():
    print("\n %s :\n %s" % (word, row))

words = list(model.wv.vocab)
print(words)













