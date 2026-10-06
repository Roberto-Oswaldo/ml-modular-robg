import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Scikit - Learn
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    return (
        ColumnTransformer,
        DecisionTreeRegressor,
        DummyRegressor,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        StandardScaler,
        mean_absolute_error,
        mo,
        pd,
        plt,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actividad 1 · Flujo de trabajo de Machine Learning

    **Machine Learning y procesamiento de datos · Trabajo individual**

    **Nombre:** Roberto Oswaldo Bañuelos Galo

    Usaremos el conjunto de datos de viviendas trabajado en clase para explicar el flujo de un análisis sencillo, separar los datos correctamente y comparar modelos. Cada fila de los datos describe una zona censal, no una vivienda individual; el objetivo es predecir la variable `median_house_value`, el valor mediano de las viviendas de la zona.

    **Instrucciones**

    1. Completa las cuatro tareas de código marcadas `TODO`.
    2. Escribe tus respuestas en los espacios de las celdas Markdown y sustituye los textos **COMPLETAR** donde aparezcan. Guarda el archivo: las respuestas deben formar parte del notebook, no quedar solo en pantalla.
    3. Usa la misma partición y semilla para todos los modelos. No uses prueba para elegir variables, modelos o hiperparámetros.
    4. Entrega este `.py` con tu nombre y respuestas, incluida la tabla de resultados.
       Antes de entregar, vuelve a abrirlo y comprueba que funciona.

    El código proporcionado también debe poder explicarse con tus palabras.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Cargar y revisar los datos

    Primero inspeccionamos la estructura y los faltantes, sin buscar todavía
    relaciones que nos ayuden a elegir el modelo.

    **Responde:** ¿por qué es un problema de regresión? Escribe dos variables
    predictoras y la variable objetivo. ¿Qué significaría una predicción para
    una fila? ¿Sería correcto interpretarla como el precio de una casa específica?

    **Respuesta:** Es regresión porque el objetivo, median_house_value, es un valor numérico continuo en dólares. Dos predictoras son median_income y housing_median_age. Cada fila es una zona censal, así que la predicción es el valor mediano estimado de las viviendas de esa zona, no el precio de una casa específica.
    """)
    return


@app.cell
def _(pd):
    # Requiere internet la carga de datos
    url= "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv"
    datos_originales = pd.read_csv(url)
    datos_originales.head()
    return (datos_originales,)


@app.cell
def _(datos_originales, mo):
    mo.vstack([
        mo.md(f"**Filas:** {len(datos_originales):,} · **Columnas:** {datos_originales.shape[1]} (incluyen el objetivo)"),
        datos_originales.dtypes.rename("tipo").to_frame().join(
            datos_originales.isna().sum().rename("faltantes")
        ),
    ])
    return


@app.cell
def _(datos_originales, mo):
    # Para esta primera actividad restringimos el análisis a registros completos.
    # Quitamos filas con datos nulos

    datos = datos_originales.dropna().copy()
    filas_eliminadas = len(datos_originales) - len(datos)
    mo.md(f"Se eliminaron **{filas_eliminadas} filas** ({filas_eliminadas / len(datos_originales):.2%}).")
    return (datos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Responde:** ¿qué columna tiene faltantes? ¿Qué porcentaje de filas eliminamos?
    ¿Por qué eliminar filas puede cambiar la población que representa el análisis?
    Propón una alternativa y explica con qué conjunto calcularías sus parámetros.

    **Respuesta:** La columna con faltantes es total_bedrooms; se eliminaron 207 filas Eliminar filas puede cambiar la población representada si los faltantes no son aleatorios. Una alternativa es imputar con la mediana, calculada solo con el conjunto de entrenamiento y aplicada a validación y prueba
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Separar antes de aprender de los datos

    **Tarea de código:** completa la segunda llamada a `train_test_split` para obtener aproximadamente **60% entrenamiento, 20% validación y 20% prueba** del conjunto de registros completos.

    Ya reservamos el 20% para prueba. ¿Qué fracción del 80% restante debes asignar a validación? Sustituye los cuatro `None` por una llamada que use `X_desarrollo`, `y_desarrollo`, `test_size=...` y `random_state=42`.

    **Antes de ejecutar, responde:**

    - ¿Qué se aprende con entrenamiento y qué decisiones se toman con validación?con entrenamiento el modelo aprende sus parámetros; con validación se decide qué modelo o configuración usar.
    - ¿Para qué se reserva prueba y por qué no debemos consultarla repetidamente?
    - ¿Por qué ajustar `StandardScaler` con todos los datos antes de dividirlos
      permite que validación influya en el análisis, aunque no usemos sus etiquetas?

    **Respuesta:** Entrenamiento y validación: con entrenamiento el modelo aprende sus parámetros; con validación se decide qué modelo o configuración usar.
    Prueba: se reserva para una evaluación final imparcial. Si la consultamos varias veces para decidir, deja de ser independiente y su resultado queda demasiado optimista.
    StandardScaler: si se ajusta antes de dividir, la media y la desviación incluyen datos de validación, así que esa información influye en el preprocesamiento (fuga de información) aunque no se usen sus etiquetas.
    """)
    return


@app.cell
def _(datos, train_test_split):
    X = datos.drop(columns="median_house_value")
    y = datos["median_house_value"]
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    return X, X_desarrollo, X_test, y_desarrollo, y_test


@app.cell
def _(X_desarrollo, train_test_split, y_desarrollo):
    # TODO 1: sustituye esta asignación por train_test_split(...).
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=0.25, random_state=42
    )
    return X_train, X_valid, y_train, y_valid


@app.cell
def _(X, X_test, X_train, X_valid, mo, pd, y_train, y_valid):
    mo.stop(X_train is None, mo.md("⏸ Completa la tarea 1 para continuar."))
    assert X_train.index.equals(y_train.index)
    assert X_valid.index.equals(y_valid.index)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)
    assert abs(len(X_train) / len(X) - 0.60) < 0.01, "Revisa la proporción de entrenamiento."
    particion_lista = True
    pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Validación", "Prueba"],
        "Filas": [len(X_train), len(X_valid), len(X_test)],
        "Fracción": [len(X_train)/len(X), len(X_valid)/len(X), len(X_test)/len(X)],
    })
    return (particion_lista,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Explorar entrenamiento y preparar las variables

    **Tarea de código:** completa la función para crear `rooms_per_household` dividiendo `total_rooms` entre `households`. Conserva las columnas originales. Se utiliza `.replace(0, 1)` en el denominador como regla explícita para evitar una división entre cero.

    Una función permite aplicar la misma operación a cada conjunto. Estas divisiones por fila no estiman parámetros a partir de otras observaciones.
    """)
    return


@app.cell
def _crear_variables():
    def _():
        def crear_variables(tabla):
            resultado = tabla.copy()
            resultado["bedrooms_per_household"] = (
                resultado["total_bedrooms"] / resultado["households"].replace(0, 1)
            )
            resultado["rooms_per_household"] = (
                resultado["total_rooms"] / resultado["households"].replace(0, 1)
            )
            return resultado
        return


    _()
    return


@app.function
def crear_variables(tabla):
    resultado = tabla.copy()
    resultado["bedrooms_per_household"] = (
        resultado["total_bedrooms"] / resultado["households"].replace(0, 1)
    )
    resultado["rooms_per_household"] = (
        resultado["total_rooms"] / resultado["households"].replace(0, 1)
    )
    return resultado


@app.cell
def _(X_train, X_valid, mo, particion_lista):
    # Espera a que la celda de comprobación valide la partición.
    mo.stop(not particion_lista)
    X_train_feat = crear_variables(X_train)
    X_valid_feat = crear_variables(X_valid)
    mo.stop("rooms_per_household" not in X_train_feat,
            mo.md("⏸ Completa la tarea 2 para continuar."))
    X_train_feat.head()
    return X_train_feat, X_valid_feat


@app.cell
def _(X_train, X_train_feat):
    print("Ceros en households:", (X_train["households"] == 0).sum())
    X_train_feat.nlargest(5, "rooms_per_household")[
        ["total_rooms", "households", "rooms_per_household"]
    ]
    return


@app.cell
def _(X_train_feat, plt, y_train):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.scatter(X_train_feat["rooms_per_household"], y_train, alpha=0.15, s=8)
    _ax.set(xlabel="Habitaciones por hogar", ylabel="Valor mediano (dólares)",
            title="Exploración del conjunto de entrenamiento")
    _fig
    return


@app.cell
def _(X_train, X_train_feat):
    print("Ceros en households:", (X_train["households"] == 0).sum())
    X_train_feat.nlargest(5, "rooms_per_household")[
        ["total_rooms", "households", "rooms_per_household"]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Responde:**

    - ¿Qué aporta una razón por hogar frente a un total?
    - Identifica valores extremos en la gráfica y propón una posible explicación. Inspecciona las filas correspondientes del conjunto de entrenamiento: ¿qué observas en `total_rooms` y `households`?
    - ¿Hay valores cero en `households` en entrenamiento? ¿Qué implicaría sustituir cero por uno al calcular las razones? Explica una limitación de esta regla, aunque no encuentres ceros.
    - ¿Por qué aplicamos exactamente la misma función a entrenamiento y validación?


    **Respuesta:** Razón vs. total: una razón compara zonas de distinto tamaño; un total depende de cuántos hogares hay y mezcla el tamaño de la zona con el tipo de vivienda.
    Valores extremos: hay puntos con muchas habitaciones por hogar (valores muy altos en el eje x). Al inspeccionarlos, tienen total_rooms alto con muy pocos households, lo que puede deberse a zonas con pocos hogares o con viviendas no habitadas de forma permanente (por ejemplo, vacacionales).
    Ceros en households: en entrenamiento hay 0 ceros. Sustituir 0 por 1 trata una zona sin hogares como si tuviera uno, lo que es una regla arbitraria que distorsiona la razón. Aunque no haya ceros, la limitación es que podría ocultar zonas con datos erróneos o vacías.
    Misma función en entrenamiento y validación: para que las variables se calculen igual en ambos conjuntos y el modelo reciba datos consistentes. Como la operación es por fila y no aprende de otras observaciones, no hay fuga de información.

    El preprocesamiento siguiente estandariza las variables numéricas y codifica `ocean_proximity`. `Pipeline` une estas operaciones con el modelo: al llamar a `fit(X_train_feat, y_train)`, todo se ajusta solo con entrenamiento. Al llamar a `predict`, se reutilizan esas transformaciones.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    X_train_feat,
):
    columnas_numericas = X_train_feat.select_dtypes(include="number").columns.tolist()

    def construir_pipeline(estimador):
        # Cada entrenamiento recibe un preprocesador nuevo.
        preprocesador = ColumnTransformer([
            ("numericas", StandardScaler(), columnas_numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["ocean_proximity"]),
        ])
        return Pipeline([("preprocesamiento", preprocesador), ("modelo", estimador)])

    return (construir_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Elegir y entrenar con un botón

    **Tarea de código:** agrega un tercer modelo al diccionario siguiendo el patrón proporcionado: un `DecisionTreeRegressor(max_depth=6, random_state=42)` con la etiqueta `Árbol de decisión`.

    El selector está envuelto en `.form()`: cambiar la selección no envía todavía su valor a Python. El botón **Entrenar** confirma la elección. El entrenamiento lee únicamente `formulario_modelo.value` y espera mientras sea `None`.

    Prueba la automatización: entrena regresión lineal, cambia el selector a bosque y observa que los resultados siguen mostrando regresión lineal. Pulsa **Entrenar** y comprueba que ahora muestran bosque. Si editas código o datos de los que depende el entrenamiento, marimo sí puede volver a ejecutar las celdas: el formulario controla los cambios del selector.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, LinearRegression, RandomForestRegressor):
    modelos = {
        "Regresión lineal": LinearRegression(),
        "Bosque aleatorio": RandomForestRegressor(n_estimators=100, random_state=42),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=6, random_state=42),
    }
    return (modelos,)


@app.cell
def _(mo, modelos):
    formulario_modelo = mo.ui.dropdown(
        options=list(modelos), value="Regresión lineal", label="Modelo",
    ).form(submit_button_label="Entrenar", clear_on_submit=False)
    formulario_modelo
    return (formulario_modelo,)


@app.cell
def _(
    X_train_feat,
    X_valid_feat,
    construir_pipeline,
    formulario_modelo,
    mo,
    modelos,
    y_train,
):
    mo.stop(formulario_modelo.value is None,
            mo.md("Selecciona un modelo y pulsa **Entrenar**."))
    nombre_entrenado = formulario_modelo.value
    modelo_entrenado = construir_pipeline(modelos[nombre_entrenado])
    modelo_entrenado.fit(X_train_feat, y_train)
    pred_train = modelo_entrenado.predict(X_train_feat)
    pred_valid = modelo_entrenado.predict(X_valid_feat)
    return nombre_entrenado, pred_train, pred_valid


@app.cell
def _(
    DummyRegressor,
    X_train_feat,
    X_valid_feat,
    mean_absolute_error,
    mo,
    nombre_entrenado,
    pd,
    pred_train,
    pred_valid,
    root_mean_squared_error,
    y_train,
    y_valid,
):
    _base = DummyRegressor(strategy="median")
    _base.fit(X_train_feat, y_train)
    _pred_base = _base.predict(X_valid_feat)
    mo.vstack([
        mo.md(f"**Último modelo entrenado: {nombre_entrenado}**"),
        pd.DataFrame({
            "Evaluación": ["Referencia: mediana · validación", "Modelo · entrenamiento", "Modelo · validación"],
            "MAE": [mean_absolute_error(y_valid, _pred_base),
                    mean_absolute_error(y_train, pred_train), mean_absolute_error(y_valid, pred_valid)],
            "RMSE": [root_mean_squared_error(y_valid, _pred_base),
                     root_mean_squared_error(y_train, pred_train), root_mean_squared_error(y_valid, pred_valid)],
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Comparar e interpretar

    Ejecuta los tres modelos y transcribe sus resultados. La salida del widget muestra la última ejecución; esta tabla conserva tu comparación en el archivo. Usaremos **MAE de validación** como criterio de selección.

    | Modelo | MAE entrenamiento | MAE validación | RMSE validación |
    |---|---|---|---|
    | Referencia: mediana | No requerido | 86,833.32 | 117,527.45 |
    | Regresión lineal | 	49,195.14 | 50,383.50 | 71,859.24 |
    | Árbol de decisión | 45,788.84 |48,081.18 | 68,629.32 |
    | Bosque aleatorio | 12,479.80 | 33,198.46 | 50,191.03 |

    **Responde:**

    1. Interpreta el MAE de un modelo en dólares y en el contexto de estas zonas.
    2. ¿Cuánto reduce el MAE frente a predecir siempre la mediana de entrenamiento?
    3. ¿Qué sugiere un error de entrenamiento mucho menor que el de validación?
    4. Elige un modelo usando la tabla y justifica tu elección antes de abrir prueba.
    5. Explica qué hacen `fit` y `predict`, y por qué cambiar el selector no entrena
       hasta pulsar el botón. Describe lo que observaste al comprobarlo.

    **Respuestas:**
    El MAE del bosque es de unos 33,198 dólares: en promedio, la predicción del valor mediano de una zona se desvía esa cantidad del valor real, algo notable frente a valores típicos de cientos de miles de dólares.
    Frente a la mediana (86,833), el bosque reduce el MAE en aproximadamente 61.8% (la regresión lineal 42.0% y el árbol 44.6%).
    Sugiere sobreajuste: el modelo memoriza el entrenamiento y generaliza peor. Pasa sobre todo con el bosque (12,480 vs. 33,198).
    Elijo el bosque aleatorio, porque tiene el menor MAE de validación (33,198 vs. 48,081 del árbol y 50,384 de la regresión lineal). Aunque sobreajusta más, sigue siendo el mejor en datos no vistos.
    fit ajusta los parámetros del modelo con entrenamiento; predict aplica lo aprendido a datos nuevos. Cambiar el selector no entrena porque .form() solo envía el valor a Python al pulsar Entrenar. Al comprobarlo, tras cambiar a bosque los resultados siguieron mostrando regresión lineal hasta pulsar el botón.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluación final en prueba

    **Tarea de código:** después de escribir tu justificación, sustituye `None` por la etiqueta exacta del modelo elegido. Esta elección es independiente del selector anterior. Después pulsa **Evaluar en prueba**.

    Para mantener sencillo el ejercicio, ajustaremos un pipeline nuevo únicamente con entrenamiento. No incorporaremos validación al ajuste final en esta tarea. No cambies la elección después de ver prueba: sus resultados se reportan, no se utilizan para seguir buscando el mejor modelo.
    """)
    return


@app.cell
def _():
    eleccion_final = "Bosque aleatorio"
    return (eleccion_final,)


@app.cell
def _(eleccion_final, mo, modelos):
    mo.stop(eleccion_final is None, mo.md("⏸ Completa la comparación y la tarea 4."))
    assert eleccion_final in modelos, "Usa una etiqueta del diccionario modelos."
    confirmar_prueba = mo.ui.checkbox(
        label=f"Ya justifiqué mi elección: {eleccion_final}",
    ).form(submit_button_label="Evaluar en prueba")
    confirmar_prueba
    return (confirmar_prueba,)


@app.cell
def _(
    X_test,
    X_train_feat,
    confirmar_prueba,
    construir_pipeline,
    eleccion_final,
    mean_absolute_error,
    mo,
    modelos,
    root_mean_squared_error,
    y_test,
    y_train,
):
    mo.stop(confirmar_prueba.value is not True,
            mo.md("Prueba permanece reservada. Confirma tu elección para evaluarla."))
    _final = construir_pipeline(modelos[eleccion_final])
    _final.fit(X_train_feat, y_train)
    _pred_test = _final.predict(crear_variables(X_test))
    mo.md(f"""
    **Evaluación final: {eleccion_final}**

    - MAE de prueba: **{mean_absolute_error(y_test, _pred_test):,.2f} dólares**.
    - RMSE de prueba: **{root_mean_squared_error(y_test, _pred_test):,.2f} dólares**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reporte final:** modelo elegido: Bosque aleatorio · MAE prueba: 33,697.54 · RMSE prueba: 50,964.58.

    **Responde:** ¿cómo se compara el error de prueba con el de validación? ¿Por qué no tienen que coincidir? Si el error de prueba es mayor, ¿por qué cambiar repetidamente de modelo mirando prueba dejaría de ser una evaluación final?

    **Respuesta:** El error de prueba (MAE 33,697.54) es muy parecido al de validación (33,198.46), solo un 1.5% mayor. No tienen que coincidir porque son muestras distintas de zonas, y cada una da una estimación con algo de variación. Si cambiáramos de modelo mirando prueba, la estaríamos usando para elegir, y dejaría de medir el desempeño con datos realmente no vistos: su resultado se volvería optimista.

    **Explica el análisis completo** en 6–8 oraciones: carga y revisión, separación, exploración, creación de variables, preprocesamiento, entrenamiento, selección y evaluación final. En cada paso indica para qué sirve y qué conjunto utiliza.

    **Explicación:** Primero cargamos los datos y revisamos tipos y faltantes; eliminamos las filas incompletas (aprox. 1%). Después separamos los datos en 60% entrenamiento, 20% validación y 20% prueba, con la misma semilla. Exploramos solo el conjunto de entrenamiento y creamos variables por fila (rooms_per_household y bedrooms_per_household), aplicando la misma función a todos los conjuntos. El preprocesamiento (StandardScaler y OneHotEncoder) va dentro de un Pipeline, de modo que se ajusta solo con entrenamiento y se reutiliza en validación y prueba. Entrenamos tres modelos con entrenamiento y los comparamos con validación mediante el MAE, junto con una referencia que predice la mediana. El bosque aleatorio tuvo el menor MAE de validación, así que lo elegimos antes de abrir prueba. Finalmente, ajustamos un pipeline nuevo con entrenamiento y lo evaluamos una sola vez en prueba, obteniendo un MAE de 33,697.54 y un RMSE de 50,964.58 dólares.

    ## Para una clase posterior · Ubicación

    No es requisito de esta entrega. Propón una variable derivada de `latitude` y `longitude`: por ejemplo, distancia a un punto de referencia fijo. ¿Qué hipótesis representa? ¿Cómo compararías con y sin ella usando validación? Si defines el punto o agrupas zonas aprendiendo de los datos, ¿con qué conjunto debes hacerlo? ¿Una división aleatoria mide necesariamente el desempeño en
    regiones geográficas completamente nuevas?

    ## Criterios de evaluación

    | Criterio | Puntos |
    |---|---:|
    | Separación correcta y explicación de entrenamiento, validación y prueba | 25 |
    | Variable nueva, revisión de datos y explicación del preprocesamiento | 20 |
    | Tercer modelo y comprobación del selector con botón | 20 |
    | Comparación, interpretación y reporte final de métricas | 25 |
    | Explicación del flujo completo y notebook guardado con respuestas | 10 |

    **Antes de entregar:** comprueba los cuatro TODO, las respuestas, la tabla y
    el reporte final. Los errores pequeños no dan más puntos por sí solos:
    se evalúan el procedimiento y la interpretación.

    **Consulta:** [formularios de marimo](https://docs.marimo.io/api/inputs/form/)
    · [fuga de información y buenas prácticas](https://scikit-learn.org/stable/common_pitfalls.html).
    """)
    return


if __name__ == "__main__":
    app.run()
