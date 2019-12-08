# -- coding: latin-1 --
"""
Elaborado por:
    Miguel Delgado
    Paula Aguirre
    Fabian Palma
    
Licencia GPLv3 
"""
#cargar librerias
        
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from gensim import models
from nltk.tokenize import word_tokenize

#Funcion de carga de los datos y su limpieza

def iniciar_datos():
    #Leyendo e csv a un data frame
    df = pd.read_csv('df_EmpleoPublico.csv', encoding="utf8")
    
    #Conviertiendo los nan a vacio para evitar problemas de parseo
    df = df.replace(np.nan, '', regex=True)
    
    #Reemplazar letras dentro del Data frame
    df = df.replace("\xc3\xa1","á")
    df = df.replace("\xc3\xa9","é")
    df = df.replace("\xc3\xad","í")
    df = df.replace("\xc3\xb3","ó")
    df = df.replace("\xc3\xba","ú")
    df = df.replace("\xc3\x81","Á")
    df = df.replace("\xc3\x89","É")
    df = df.replace("\xc3\x8d","Í")
    df = df.replace("\xc3\x93","Ó")
    df = df.replace("\xc3\x9a","Ú")
    df = df.replace("\xc3±","ñ")
    
        
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
  
    # Renombrar Variables.
    df.columns =['18', 'Cargo', 'Ciudad', 'Condiciones', 'd_abierto', 'd_proceso',
                  'd_selesccion', 'descripcion', 'Diurno_x', 
                  'Honorarios Suma Alzada_x', 'IV Turno_x', 'institucion', 
                  'mes_x', 'ministerio', 'n_vacantes', 'obj_del_cargo', 
                  'page_id', 'region_x', 'region', 'renta_bruta', 'requesitos','suplencia',
                  'tareas','t_vacante', 'ano','a_trabajo', 'c_resultado_a', 'c_resultado_b', 'salario', 'busqueda']
  
    
    #Dimensiones del df
    df.shape 
    # (44554, 29)
    
    # Cuenta nulos
    df.isnull().sum()
    #18                          44552 nulos de 44554      --> No aporta
    #Cargo                         324
    #Ciudad                        335
    #Condiciones                 44552 nulos de 44554      --> No aporta
    #d_abierto                    2312
    #d_proceso                    2312
    #d_selesccion                 2312
    #descripcion                 44553 nulos de 44554      --> No aporta
    #Diurno_x                    44553 nulos de 44554      --> No aporta
    #Honorarios Suma Alzada_x    44553 nulos de 44554      --> No aporta
    #IV Turno_x                  44553 nulos de 44554      --> No aporta
    #institución                     0
    #mes_x                        2312
    #ministerio                      0
    #n_vacantes                      0
    #obj_del_cargo                3260
    #page_id                        0
    #region_x                    44552 nulos de 44554      --> No aporta
    #region                      337
    #renta_bruta                 17679 nulos de 44554      --> Aporte Bajo
    #requesitos                  44551 nulos de 44554      --> No aporta
    #suplencia                   44553 nulos de 44554      --> No aporta
    #tareas                      44553 nulos de 44554      --> No aporta
    #t_vacante                     335
    #ano                          2312
    #a_trabajo                     343
    #c_resultado _a                  5
    #c_resultado_b                   0
    #salario                    17679 nulos de 44554      --> Aporte Bajo
    
    
    # Selección de Variables Relevantes
    df = df[['Ciudad','Cargo','d_abierto', 'd_proceso','d_selesccion', 'institucion', 'ministerio','n_vacantes', 
             'obj_del_cargo', 'region','t_vacante','a_trabajo', 'c_resultado_a','c_resultado_b','salario','renta_bruta',
             'ano','mes_x','page_id', 'busqueda']]
    
    #Conviertiendo los nan a vacio para evitar problemas de parseo
    df = df.replace(np.nan, '', regex=True)
    
    #Mostrando los primeros registros
    data = df.values
    print(data)
    
    return df

#Funcion de creaación y utilización del modelo
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

# Funcion que hace los cruces entre la palabra ingresada por el usuario y
#las palabras claves de los trabajos de la bd
def buscar(df, model, querysearch):
    #obteniendo los tags (cargos)
        
    try:
        modelsearch = model[querysearch]
    except:
        print('Lo sentimos, no hay resultados, intente con algo mas amplio')
        return 0
    resultados = model.docvecs.most_similar(positive=modelsearch)
    
    for resultado in resultados:
        salida = []
        if resultado[1]>=UMBRAL:
            salida =  df[df['Cargo'] == resultado[0]]
            for index, row in salida.iterrows():
                print('*** Oportunidad****')
                print('Cargo: ', row['Cargo'])
                print('Ciudad: ', row['Ciudad'])
                print('Salario: ', row['salario'])        
                print('Institucion: ', row['institucion'])
                print('Ministerio: ', row['ministerio'])
                print('No de Vacantes: ', row['n_vacantes'])        
                print('Región: ', row['region'])        
                print('*******')        
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
        
#codigo para solicitar palabra de busqueda del usuario 
       
while True:
    q = input("Ingrese su palabra o palabras de busqueda de empleo:  (escriba exit para salir) y presione enter \n ")
    if q == 'exit':
        break
    qtoken =  (q.lower().split())
    t = buscar(df, model, qtoken)
print("Gracias por venir, vuelva pronto")