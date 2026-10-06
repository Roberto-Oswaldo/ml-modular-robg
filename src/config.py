from pathlib import Path

# Rutas
RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data" / "housing.csv"
URL_DATOS = (
    "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/"
    "refs/heads/main/data/housing/housing.csv"
)

# Variable objetivo
VARIABLE_OBJETIVO = "median_house_value"

# Variables predictoras
COLUMNAS_NUMERICAS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
COLUMNAS_CATEGORICAS = ["ocean_proximity"]
PREDICTORES = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS

# Semilla y proporciones de la división (60% entrenamiento, 20% validación, 20% prueba)
SEMILLA = 42
TAMANO_PRUEBA = 0.20
TAMANO_VALIDACION = 0.25  # 25% del 80% restante = 20% del total