from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the available drug types
drugs = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

# Define the pattern for different frequency model files
model_file_patterns = {
    'daily': 'auto_arima_model_daily_{drug}.pkl',
    'weekly': 'auto_arima_model_week_{drug}.pkl',
    'monthly': 'auto_arima_model_{drug}.pkl'
}

# Path to the models folder
models_folder = 'models'

# Function to load the appropriate model based on the frequency and drug type
def load_model(freq, drug):
    model_file = model_file_patterns[freq].format(drug=drug)
    model_path = os.path.join(models_folder, model_file)
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Routes for rendering pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/drug-info')
def drug_info():
    return render_template('drug.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/prediction2')
def prediction():
    return render_template('predictions.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/no5c')
def no5c():
    return render_template('no5c.html')

@app.route('/no5b')
def no5b():
    return render_template('no5b.html')

@app.route('/mo1ab')
def mo1ab():
    return render_template('mo1ab.html')

@app.route('/mo1ae')
def mo1ae():
    return render_template('mo1ae.html')

@app.route('/no2ba')
def no2ba():
    return render_template('no2ba.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/details')
def details():
    return render_template('detail.html')

@app.route('/drugs')
def drugs_info():
    return render_template('drugs.html')

@app.route('/prediction')
def prediction2():
    return render_template('prediction2.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    date = data['date']
    prediction_type = data['type']

    # Define the appropriate date range and frequency based on prediction type
    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'yearly': 'Y'
    }

    # Convert the date string to a pandas Timestamp
    target_date = pd.to_datetime(date)

    # Prepare the date range for prediction
    start_date = '2019-11-30'  # Start date for all models
    date_range = pd.date_range(start=start_date, end=target_date, freq=freq_map[prediction_type])

    predictions = {}
    for drug in drugs:
        # Load the appropriate model
        model = load_model(prediction_type, drug)
        
        # Make predictions up to the target date
        predicted_values = model.predict(n_periods=len(date_range))
        
        # Extract the prediction for the specific date (end of the range)
        predictions[drug] = predicted_values[-1]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(port=5004)
