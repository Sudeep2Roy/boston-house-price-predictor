from flask import Flask, render_template, request
import numpy as np
import pickle
import webbrowser
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home route to render form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form and convert to float
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])

        # Predict using loaded model
        prediction = model.predict(final_features)[0]

        return render_template('index.html', prediction_text=f'Estimated House Price: ${round(prediction, 2)}k')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Function to open browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# Start the app
if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True, use_reloader=False)  # use_reloader=False prevents two browser tabs
