from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src import config


def obtener_modelos():
    """Diccionario con los modelos a comparar. Todos usan la misma semilla."""
    return {
        "Regresión lineal": LinearRegression(),
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=6, random_state=config.SEMILLA
        ),
        "Bosque aleatorio": RandomForestRegressor(
            n_estimators=100, random_state=config.SEMILLA
        ),
    }


def obtener_referencia():
    """Modelo base que siempre predice la mediana del entrenamiento."""
    return DummyRegressor(strategy="median")