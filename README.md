## Classificador de Flor de Íris
Esses modelos criados com machine learning são capases de classificar flores de íris em três categorias:
- Íris setosa;
- Íris versicolor;
- Íris virginica.

Ele se utiliza de 4 categorias para fazer a classificação:
- Comprimento da sépala;
- Largura da sépala;
- Comprimento da pétala;
- Largura da pétala.

Cada um deles utiliza uma abordagem diferente para sua criação, no intuito de demonstar a diferença em acertividade de cada um deles.

### Objetivo
Esse modelo foi desenvolvido com o ituito de aplicar os conhecimentos adquiridos durante dois cursos de Aprendizado de Maquina efetuados na plataforma Kaggle, [*Intro to Machinhe Learning*](https://www.kaggle.com/learn/intro-to-machine-learning) e [*Intermediate Machine Learning*](https://www.kaggle.com/learn/intermediate-machine-learning)

### Analisando os dados
Primeiro o dataset foi análisa para a falata de entradas nas suas colunas, para válidar a necessidade de se remover ou alterar valores, porém neste caso os dados etão todos completos, como pode ser válidado rodando o seguinte código:
```python
    import pandas as pd

    path = "../Dataset/IRIS.csv"
    ds_iris = pd.read_csv(path)

    print("Quantidade de valores faltando:")
    print(ds_iris.isna().sum() + )
```
Que resulta na seguinte saida:
```cmd
    Quantidade de valores faltando:
    sepal_length    0
    sepal_width     0
    petal_length    0
    petal_width     0
    species         0
    dtype: int64
```
Assim mostrando que o dataset está completo, e pode ser usado sem grandes alteraçãoes.
### Referencias
- **DATASET**
    O *dataset* IRIS.csv foi disponibilizado na plataforma Kaggle sobre [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/), e pode ser visto neste [link](https://www.kaggle.com/datasets/arshid/iris-flower-dataset/data).