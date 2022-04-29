import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #libreria del modelo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

simplefilter(action='ignore', category=FutureWarning)

url = 'weatherAUS.csv'
data = pd.read_csv(url)

data.RainToday.replace(['No', 'Yes'], [0, 1], inplace= True)
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace= True)

data.drop(['WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'Location', 'MinTemp'
           , 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm'
           , 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm', 'Sunshine','Date', 'Evaporation'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)

# Partir la data por la mitad (Media pa training y media pa testing)

data_train = data[:71097]
data_test = data[71097:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow) # 0 No llueve hoy 1 si llueve hoy


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow) # 0 No llueve hoy 1 si llueve hoy

# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# RANDOM FOREST

rf = RandomForestClassifier()

# Entrenar el modelo
rf.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')


# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')

# K-Nearest neighbors

# Seleccionar un modelo

kn = KNeighborsClassifier()

# Entrenar el modelo

kn.fit(x_train, y_train)

# Metricas

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {kn.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {kn.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')