import pandas as pd

from src.data import cargar_datos, separar_predictores_objetivo
from src.evaluate import evaluar
from src.models import obtener_modelos, obtener_referencia
from src.preprocessing import construir_pipeline
from src.split import dividir_datos


def main():
    # 1. Cargar datos y separar predictores/objetivo
    X, y = separar_predictores_objetivo(cargar_datos())

    # 2. Dividir en entrenamiento, validación y prueba
    X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos(X, y)

    # 3. Entrenar cada modelo solo con entrenamiento y evaluarlo
    candidatos = {"Referencia: mediana": obtener_referencia()}
    candidatos.update(obtener_modelos())

    resultados = []
    entrenados = {}
    for nombre, estimador in candidatos.items():
        pipeline = construir_pipeline(estimador)
        pipeline.fit(X_train, y_train)
        entrenados[nombre] = pipeline
        m_train = evaluar(pipeline, X_train, y_train)
        m_valid = evaluar(pipeline, X_valid, y_valid)
        resultados.append({
            "Modelo": nombre,
            "MAE entrenamiento": m_train["MAE"],
            "MAE validación": m_valid["MAE"],
            "RMSE validación": m_valid["RMSE"],
        })

    tabla = pd.DataFrame(resultados)
    print("\nComparación en entrenamiento y validación:")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # 4. Elegir el mejor modelo con validación (sin contar la referencia)
    modelos_reales = tabla[tabla["Modelo"] != "Referencia: mediana"]
    mejor = modelos_reales.loc[modelos_reales["MAE validación"].idxmin(), "Modelo"]
    print(f"\nModelo elegido (menor MAE de validación): {mejor}")

    # 5. Evaluar solo el modelo elegido en prueba, una única vez
    m_test = evaluar(entrenados[mejor], X_test, y_test)
    print(f"MAE de prueba:  {m_test['MAE']:,.2f}")
    print(f"RMSE de prueba: {m_test['RMSE']:,.2f}")


if __name__ == "__main__":
    main()