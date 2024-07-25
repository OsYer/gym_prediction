from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

try:
    # Asegúrate de que los nombres de las variables sean correctos
    scaler = joblib.load('escaler.pkl')
    selector = joblib.load('selectore.pkl')
    model = joblib.load('random_forest_modelo.pkl')
except Exception as e:
    app.logger.error(f'Error al cargar los modelos: {str(e)}')

@app.route('/')
def home():
    return render_template('formulario1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        edad = float(request.form['edad'])
        total_asistencias = float(request.form['total_asistencias'])
        ingresos_estimados = float(request.form['ingresos_estimados'])
        problema_salud_reciente = int(request.form['problema_salud_reciente'])
        distancia_gimnasio_metros = float(request.form['distancia_gimnasio_metros'])
        num_renovaciones_anual = float(request.form['num_renovaciones_anual'])
        
        # Crear un DataFrame con todas las características necesarias
        data_df = pd.DataFrame([[
            edad, total_asistencias, 0, 0, ingresos_estimados, problema_salud_reciente,
            distancia_gimnasio_metros, num_renovaciones_anual, 0, 0
        ]], columns=[
            'edad', 'total_asistencias', 'estudia', 'trabaja', 'ingresos_estimados',
            'problema_salud_reciente', 'distancia_gimnasio_metros', 'num_renovaciones_anual',
            'renovo_anterior', 'genero_M'
        ])
        
        # Normalizar los datos
        data_scaled = scaler.transform(data_df)
        
        # Seleccionar características
        data_rfe = selector.transform(data_scaled)

        # Realizar la predicción
        prediction = model.predict(data_rfe)
        respuesta = int(prediction[0])
        
        return jsonify({'renueva_membresia': respuesta})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
