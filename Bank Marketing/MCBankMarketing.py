import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #libreria del modelo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)

url = 'bank-full.csv'
data = pd.read_csv(url)

#Tratamiento de los datos

data.marital.replace(['married', 'single', 'divorced'], [2, 1, 0], inplace= True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace= True)
data.default.replace(['no', 'yes'], [0, 1], inplace= True)
data.housing.replace(['no', 'yes'], [0, 1], inplace= True)
data.loan.replace(['no', 'yes'], [0, 1], inplace= True)
data.y.replace(['no', 'yes'], [0, 1], inplace= True)
data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace= True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace= True)

data.drop(['balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'job'], axis=1, inplace=True)
data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0, how='any', inplace=True)

# Partir la data en dos

data_train = data[:22605]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) # 0 desconocido 1 fallo 2 otro 3 exito

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_train.drop(['y'], 1))
y_test_out = np.array(data_train.y) # 0 desconocido 1 fallo 2 otro 3 exito


