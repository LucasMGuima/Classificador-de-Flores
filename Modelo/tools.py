from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def get_score(X, y, n_estimator, my_cv=5) -> float:
    """
        Retorna uma média do MAE sobre a quantidade desajada de CV,
        por pardão é 5, em um modelo random frest

        Parametros:\n
        X -- valores para fazer a previsao\n
        y -- valores de reusltado da previsao\n
        n_estimator -- o número de arvores no modelo\n
        cv -- a quantidade de modelos a serem feitos
    """

    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimator, random_state=0))
    ])

    score = -1 * cross_val_score(pipeline, X, y, cv=my_cv, scoring="neg_mean_absolute_error")
    return score.mean()
