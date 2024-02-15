import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tools import get_score

#Primeiro vamos importar e analizar os dados
path = "../Dataset/IRIS.csv"
ds_iris = pd.read_csv(path)

#Atribuimos a X as caracteristicas e a y os tipos
y = ds_iris.species
y_ordered = y.copy()

species = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica': 2}
for i in y.index:
    y_ordered[i] = species[y[i]]

filter = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = ds_iris[filter]

mae = get_score(X, y_ordered, 100)

print("O Erro Absoluto Médio deste modelo é: {:.5f}".format(mae))