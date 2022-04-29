import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #libreria del modelo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)

url = 'diabetes.csv'
data = pd.read_csv(url)

#Tratamiento de los datos

data.Age.replace(np.nan, 33, inplace=True)
rangos = [20, 35, 50, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.drop(['DiabetesPedigreeFunction', 'BMI', 'Insulin', 'BloodPressure'], axis=1, inplace=True)

# Partir la data por la mitad (Media pa training y media pa testing)

data_train = data[:384]
data_test = data[384:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 

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
