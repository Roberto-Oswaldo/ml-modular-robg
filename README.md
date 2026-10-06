# ml-modular-robg

Proyecto modular de Machine Learning para predecir el valor mediano de las viviendas en zonas censales de California. Es la versión modular, en varios archivos de Python, de un notebook de clase (`notebooks/actividad_01_housing_Bañuelos_Roberto.py`).

## Problema y datos

Es un problema de **regresión**: se predice `median_house_value` (valor mediano de las viviendas de una zona, en dólares). Cada fila del dataset es una zona censal, no una vivienda individual. El dataset de viviendas se usó en clase y tiene 20,640 filas; al quitar las 207 filas con datos faltantes en `total_bedrooms` quedan 20,433.

Variables predictoras: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` y `ocean_proximity` (categórica).

## Integrantes

| Nombre | Usuario de GitHub |
|---|---|
| Roberto Oswaldo Bañuelos Galo | [Roberto-Oswaldo](https://github.com/Roberto-Oswaldo) |

Trabajo individual.

## Organización de los archivos

```
ml-modular-robg/
├── data/                  # housing.csv (no se sube a GitHub)
├── notebooks/             # notebook original de referencia
├── src/
│   ├── config.py          # rutas, variables, objetivo, semilla y proporciones
│   ├── data.py            # carga de datos y separación de X e y
│   ├── split.py           # división en entrenamiento, validación y prueba
│   ├── preprocessing.py   # variables nuevas y pipeline de preprocesamiento
│   ├── models.py          # modelos a comparar y modelo de referencia
│   └── evaluate.py        # métricas MAE y RMSE
├── train.py               # archivo principal que coordina el flujo
├── requirements.txt
└── README.md
```

## Instalación y datos

1. Clonar el repositorio y entrar a la carpeta:

```
git clone https://github.com/Roberto-Oswaldo/ml-modular-robg.git
cd ml-modular-robg
```

2. Crear y activar un entorno virtual (en Windows):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Descargar los datos a la carpeta `data`:

```
mkdir data
curl -o data\housing.csv https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv
```

Si `data/housing.csv` no existe, el programa lo descarga directamente de esa dirección (requiere internet).

## Ejecución

```
python train.py
```

El script carga los datos, los divide (60% entrenamiento, 20% validación, 20% prueba, semilla 42), entrena cada modelo solo con entrenamiento, los compara en validación, elige el de menor MAE de validación y lo evalúa una sola vez en prueba.

## Modelos y resultados

El preprocesamiento (creación de `rooms_per_household` y `bedrooms_per_household`, `StandardScaler` y `OneHotEncoder`) va dentro de un `Pipeline` junto con el modelo, así que se ajusta solo con entrenamiento.

| Modelo | MAE entrenamiento | MAE validación | RMSE validación |
|---|---|---|---|
| Referencia: mediana | 88,597.14 | 86,833.32 | 117,527.45 |
| Regresión lineal | 49,195.14 | 50,383.50 | 71,859.24 |
| Árbol de decisión | 45,788.84 | 48,081.18 | 68,629.32 |
| Bosque aleatorio | 12,479.80 | 33,198.46 | 50,191.03 |

Modelo elegido con validación: **Bosque aleatorio**. Resultado en prueba: **MAE 33,697.54** y **RMSE 50,964.58** dólares.

**Interpretación:** el bosque se equivoca en promedio unos 33,700 dólares al estimar el valor mediano de una zona, y reduce el error de la referencia en más de 60%. El MAE de prueba es muy parecido al de validación, lo que indica que la comparación fue consistente. Su error de entrenamiento es mucho menor que el de validación, es decir, sobreajusta, aunque sigue siendo el mejor con datos no vistos.

## Limitaciones y problemas conocidos

- Se eliminan las filas con datos faltantes, lo que puede cambiar un poco la población que representa el análisis.
- No se ajustaron hiperparámetros; los modelos usan valores fijos.
- El modelo final se ajusta solo con entrenamiento, sin incorporar validación.
- La división es aleatoria, así que no mide el desempeño en regiones geográficas completamente nuevas.
- Las predicciones son por zona censal, no por vivienda individual.

## Tabla de contribuciones

| Contribución | Archivo | Pull request | Commit |
|---|---|---|---|
| Configuración | `src/config.py` | #1 | 77f98fb |
| Carga de datos | `src/data.py` | #2 | 9bcc971 |
| División de datos | `src/split.py` | #3 | 9708b11 |
| Preprocesamiento | `src/preprocessing.py` | #4 | 5df5b62 |
| Modelos | `src/models.py` | #5 | 5845de4 |
| Evaluación | `src/evaluate.py` | #6 | ec35341 |
| Archivo principal | `train.py` | #7 | b9c6e68 |
| Dependencias | `requirements.txt`, `.gitignore` | #8 | 5de288b |

## Reflexion final

¿Qué implementaste y qué decisión técnica tomaste?
Reorganicé mi notebook de viviendas en un proyecto modular: configuración, carga, división, preprocesamiento, modelos, evaluación y un train.py que coordina todo. La decisión principal fue meter crear_variables dentro del Pipeline junto con el escalado y la codificación. Así todo se ajusta solo con entrenamiento y las mismas transformaciones se aplican solas a validación y prueba, sin riesgo de fuga de información ni de olvidar aplicarlas.

¿Cómo verificaste tu aportación?
Probé cada módulo con un comando corto antes de subirlo (por ejemplo, que la división diera 12259, 4087 y 4087 filas). Al final comparé la salida de train.py con mi notebook original y coincidió exactamente (MAE de prueba 33,697.54). También cloné el repo en una carpeta limpia y seguí mi README para confirmar que corría desde cero.

¿Qué observaste o aprendiste al revisar el trabajo de otra persona?
lo hice por mi propia cuenta aprendi que llevaba cada codigo por el repositorio del profesor.

¿Qué mejorarías en la siguiente versión?
Ajustaría hiperparámetros con validación, porque ahora usan valores fijos. También ajustaría el modelo final con entrenamiento y validación juntos antes de evaluar en prueba, agregaría pruebas automáticas a cada módulo y probaría una división por regiones geográficas, ya que la aleatoria no mide el desempeño en zonas completamente nuevas.
