from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


datapd = pd.read_csv('heart_failure_dataset_procesado.csv')

def limpieza_categorizacion_datos(data):
    # Verificar valores faltantes
    faltantes = data.isnull().sum().sum()
    if faltantes > 0:
        print(f"¡Hay {faltantes} datos faltantes en el DataFrame!")
    else:
        print("No hay datos faltantes en el DataFrame.")

    # Verificar filas duplicadas
    duplicados = data.duplicated().sum()
    if duplicados > 0:
        print(f"Hay {duplicados} filas duplicadas en el DataFrame.")
        data.drop_duplicates(inplace=True)
    else:
        print("No hay filas duplicadas en el DataFrame.")

    # Verificar y eliminar valores atípicos
    for column in data.select_dtypes(include='number').columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        filtro_sin_atipicos = (data[column] >= q1 - 1.5 * iqr) & (data[column] <= q3 + 1.5 * iqr)
        data = data[filtro_sin_atipicos]

    return data

data_limpia=limpieza_categorizacion_datos(datapd)

eliminar_columnas = ['DEATH_EVENT', 'age', 'Edad_Categoria']
X = data_limpia.drop(eliminar_columnas, axis=1)
y = data_limpia['age']

modelo_regresion = LinearRegression()
modelo_regresion.fit(X, y)


edades_predichas = modelo_regresion.predict(X)

mse = mean_squared_error(y, edades_predichas)
print(f"El error cuadrático medio es: {mse}")
