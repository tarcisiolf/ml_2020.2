#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

#%%
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import numpy as np

print("Current working directory: \t", os.getcwd())

os.chdir("/home/tarcisio/Documentos/10_periodo/aprendizagem_de_maquina/ml_2020.2/task1")
print("Current working directory: \t", os.getcwd())
#%%
print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
initial_data = pd.read_csv('diabetes_dataset.csv')
#%% 
def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())
#%% Pré-Processamento de Dados 1
# Aplicando a média da coluna nos valores Nan
data = data.apply(lambda x: x.fillna(x.mean()),axis=0)

#Normalizando os valores do conjunto de dados
data = minmax_norm(data)

# "accuracy":0.6377551020408163"

#%% Teste 1
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# TODO Substituir pela sua chave aqui
DEV_KEY = "MLTL"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")

#%% Pré-Processamento de Dados 2
# Gerar valores aleatórios entre o menor e o maior valor da coluna
min_values = data.min()
max_values = data.max()

a1 = data['Glucose'].isnull()
aux = np.random.randint(min_values['Glucose'], max_values['Glucose'], a1.sum())
data.loc[a1, 'Glucose'] = aux

a2 = data['BloodPressure'].isnull()
aux2 = np.random.randint(min_values['BloodPressure'], max_values['BloodPressure'], a2.sum())
data.loc[a2, 'BloodPressure'] = aux2

a3 = data['SkinThickness'].isnull()
aux3 = np.random.randint(min_values['SkinThickness'], max_values['SkinThickness'], a3.sum())
data.loc[a3, 'SkinThickness'] = aux3


a4 = data['Insulin'].isnull()
aux4 = np.random.randint(min_values['Insulin'], max_values['Insulin'], a4.sum())
data.loc[a4, 'Insulin'] = aux4

a5 = data['BMI'].isnull()
aux5 = np.random.uniform(min_values['BMI'], max_values['BMI'], a5.sum())
data.loc[a5, 'BMI'] = aux5

#Normalizando os valores do conjunto de dados
data = minmax_norm(data)

# Acuracy 0.336734693877551

#%% Teste 2
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# TODO Substituir pela sua chave aqui
DEV_KEY = "MLTL"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")


#%% Pré-Processamento de Dados 2
# Removendo a coluna "Insulin" - Utilizando média
del data['Insulin']

# "accuracy" - 0.34183673469387754
#%% Teste 2
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness'
                , 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# TODO Substituir pela sua chave aqui
DEV_KEY = "MLTL"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
#%% Pré-Processamento de Dados 3
# Removendo a coluna "Skin Thickness" - Utilizando média
del data['SkinThickness']

#%% Teste 3

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# TODO Substituir pela sua chave aqui
DEV_KEY = "MLTL"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")

