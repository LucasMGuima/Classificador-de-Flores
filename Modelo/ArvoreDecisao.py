import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

#Primeiro vamos importar e analizar os dados
path = "../Dataset/IRIS.csv"
ds_iris = pd.read_csv(path)

#Atribuimos a X as caracteristicas e a y os tipos
ordinal_encoder = OrdinalEncoder()
y = ds_iris.species
y_ordered = pd.Series()

species = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica': 2}
for i in y.index:
    y[i] = species[y[i]]


filter = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = ds_iris[filter]

#Vamos dividir os dados em dois grupos, um de treino e outro de validação
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

#Agora vamos criar e treinar o modelo
model_decisionTree = DecisionTreeRegressor()
model_decisionTree.fit(train_X, train_y)

#Por ultimo vemos como o modelo se saiu, avilando com o erro absoluto medio
prediction = model_decisionTree.predict(val_X)
mean = mean_absolute_error(val_y, prediction)

print("O Erro Absoluto Médio deste modelo é: {:.5f}".format(mean))