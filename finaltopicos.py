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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from gensim import models
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


#Funcion de carga de los datos y su limpieza

def iniciar_datos():
    #Leyendo e csv a un data frame
    df = pd.read_csv('df_EmpleoPublico.csv', encoding="latin1")
    
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
    df = df.replace({"Ã¡":"a", "Ã©":"e","Ã­":"i","Ã³":"o","Ãº":"u","Ã±":"n"}, regex=True)
    
        
    #Mostrando los nombres de las columnas
    names = df.columns.values.tolist()
    print(names)
    
    
    
    #Combinando las dos columnas del dataframe en 1 Cargo + Objetivo del Cargo -> busqueda
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
    print("Algunos Cargos unicos")
    print(cargos_unicos)
  
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


def generar_train_test():
    T = dict();  
    #Importando un csv con cargos reales y omito las lineas malas
    dftrain = pd.read_csv('muestra_cargos_reales.csv',sep = ';', encoding="utf8")
    #Train y test
    T['train'], T['test'] = train_test_split(dftrain, test_size=.30, random_state=43)
    return T
   

def entrenar_modelo(train, test):
    #definiedo los tags para train
    tagged_tr = [models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()),
                tags=[str(i)]) for i, _d in enumerate(train.cargo_real)]
    
    
    model = models.Doc2Vec(vector_size=100,
                window=5, 
                alpha=.025, 
                min_alpha=0.00025, 
                min_count=2, 
                dm=1, 
                workers=8)
    model.build_vocab(tagged_tr)
    
    #entrenando el modelo
    epochs = range(5)
    for epoch in epochs:
        print(f'Entrenado con Epoch {epoch+1}')
        model.train(tagged_tr,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # Disminuyendo al tasa de incremento
        model.alpha -= 0.00025
        # FIX tasa de aprendizaje sin decaer
        model.min_alpha = model.alpha
 
        model.save('jobs.model')
    return model




def generar_predictor(model, train, test):
    tagged_tr = [models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()),
                tags=[str(i)]) for i, _d in enumerate(train.cargo_real)]
    
    
     #Extrayendo vecores de train
    print('...Extrayendo vectores de train')
    X_train = np.array([model.docvecs[str(i)] for i in range(len(tagged_tr))])
    y_train = train['cargo_real']
    tagged_test = [models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()),
    tags=[str(i)]) for i, _d in enumerate(test.cargo_real)]
    
    print('...X train')
    print(X_train)
    print('...Y train')
    print(y_train)
    
    
    #Usando infer para inferir ls vectores de test
    print("Usando infer para inferir ls vectores de test")
    X_test = np.array([model.infer_vector(tagged_test[i][0]) for i in range(len(tagged_test))])

    
    print('... Iniciando regresion logisitca...')
    lrc = LogisticRegression(C=5, multi_class='multinomial', solver='saga',max_iter=2)
    lrc.fit(X_train,y_train)
    print('... Fitting...')
    return lrc



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
    
    #construyendo el vocabulario con la lista de docs etiquetados
    model.build_vocab(docs)
    

    #Mejorando el performance
    model.init_sims(replace=True)
    return model

# Funcion que hace los cruces entre la palabra ingresada por el usuario y
#las palabras claves de los trabajos de la bd
def buscar(df, model, querysearch, predictor):
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
                # Calculando metricas: 
                y_pred = predictor.predict(modelsearch)
                array_cargo =[row['Cargo']]
                array_ypred = [y_pred[0]]
                accuracy = accuracy_score(array_cargo, array_ypred )
                print('Predicetd % s Accuracy: %.2f' % (y_pred, accuracy) )
                print('***************')
      
    return 1

            
        

#Definiendo umbral de similitud para reportar resultados
UMBRAL = 0.34

#Iniciando el dataframe
print('Iniciando dataframe...')
df = iniciar_datos()

#Generando train y test
print('Generando data train y test')
T = generar_train_test()
train = T['train']
test = T['test']

#Entrenando el modelo para metricas
print('Entrenando modelo para metricas...')
modeltest = entrenar_modelo(train,test)

#Generando predictor
print("Predictor del modelo test para metricas: ")
predictor = generar_predictor(modeltest, train, test)

#Inicinado el modelo de busqueda
print('Iniciando modelo de busqueda...')
model = iniciar_modelo(df)

print('Bienvenido al motor avanzado de busqueda de empleo')        
#codigo para solicitar palabra de busqueda del usuario 
       
while True:
    q = input("Ingrese su palabra o palabras de busqueda de empleo:  (escriba exit para salir) y presione enter \n ")
    if q == 'exit':
        break
    qtoken =  (q.lower().split())
    t = buscar(df, model, qtoken, predictor)
print("Gracias por venir, vuelva pronto")