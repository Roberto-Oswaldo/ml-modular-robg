from sklearn.model_selection import train_test_split

from src import config


def dividir_datos(X, y):
    """Divide en entrenamiento (60%), validación (20%) y prueba (20%)."""
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=config.TAMANO_PRUEBA, random_state=config.SEMILLA
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo,
        test_size=config.TAMANO_VALIDACION, random_state=config.SEMILLA,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test