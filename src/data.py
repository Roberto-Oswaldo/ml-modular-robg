import pandas as pd

from src import config


def cargar_datos():
    """Carga el CSV (local si existe, si no desde la URL) y quita filas con nulos."""
    origen = config.RUTA_DATOS if config.RUTA_DATOS.exists() else config.URL_DATOS
    datos = pd.read_csv(origen)
    return datos.dropna().copy()


def separar_predictores_objetivo(datos):
    """Regresa X (predictores) y y (variable objetivo)."""
    X = datos[config.PREDICTORES]
    y = datos[config.VARIABLE_OBJETIVO]
    return X, y