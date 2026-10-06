from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

VARIABLES_NUEVAS = ["bedrooms_per_household", "rooms_per_household"]


def crear_variables(tabla):
    """Crea razones por hogar. Operación por fila, no aprende de otros datos."""
    resultado = tabla.copy()
    hogares = resultado["households"].replace(0, 1)
    resultado["bedrooms_per_household"] = resultado["total_bedrooms"] / hogares
    resultado["rooms_per_household"] = resultado["total_rooms"] / hogares
    return resultado


def construir_pipeline(estimador):
    """Pipeline: crear variables -> escalar/codificar -> modelo.
    Todo se ajusta solo con los datos que reciba fit (entrenamiento)."""
    preprocesador = ColumnTransformer([
        ("numericas", StandardScaler(), config.COLUMNAS_NUMERICAS + VARIABLES_NUEVAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         config.COLUMNAS_CATEGORICAS),
    ])
    return Pipeline([
        ("variables", FunctionTransformer(crear_variables)),
        ("preprocesamiento", preprocesador),
        ("modelo", estimador),
    ])