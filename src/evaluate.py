from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calcular_metricas(y_real, y_pred):
    """MAE y RMSE en dólares (problema de regresión)."""
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": root_mean_squared_error(y_real, y_pred),
    }


def evaluar(modelo, X, y):
    """Predice con un modelo ya entrenado y regresa sus métricas."""
    return calcular_metricas(y, modelo.predict(X))