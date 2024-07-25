from flask import Flask, request, jsonify, render_template
import logging
import joblib

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    random_forest_model = joblib.load('random_forest_model.pkl')
    app.logger.debug("Modelos cargados correctamente.")
except FileNotFoundError as e:
    app.logger.error(f"Error al cargar los modelos: {e}")
    raise

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        data = request.json
        app.logger.debug(f"Valores recibidos: {data}")

        # Extraer los valores de las características
        edad = int(data['edad'])
        genero = 1 if data['genero'] == 'M' else 0
        total_asistencias = int(data['total_asistencias'])
        ingresos_estimados = float(data['ingresos_estimados'])
        problema_salud_reciente = int(data['problema_salud_reciente'])
        distancia_gimnasio_metros = float(data['distancia_gimnasio_metros'])
        num_renovaciones_anual = int(data['num_renovaciones_anual'])
        estudia = int(data['estudia'])
        trabaja = int(data['trabaja'])
        renovo_anterior = int(data['renovo_anterior'])

        # Crear un array con los valores en el orden correcto
        X = [[edad, genero, total_asistencias, ingresos_estimados, problema_salud_reciente, distancia_gimnasio_metros, num_renovaciones_anual, estudia, trabaja, renovo_anterior]]

        # Aplicar el escalador
        X_scaled = scaler.transform(X)

        # Seleccionar las características relevantes
        X_selected = selector.transform(X_scaled)

        # Realizar la predicción
        prediction = random_forest_model.predict(X_selected)
        renueva = bool(prediction[0])

        return jsonify({'renueva': renueva})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
