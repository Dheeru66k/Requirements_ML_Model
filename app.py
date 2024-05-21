from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import os
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained models
vectorizer = pickle.load(open('Models/countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
grid = pickle.load(open('Models/Type_model_grid.pkl', 'rb'))

# Define a function to preprocess the input
def preprocess_input(input_text):
    if pd.isna(input_text):
        input_text = ''  # or some other placeholder or handling mechanism
    vectorized_text = vectorizer.transform([input_text])
    scaled_text = scaler.transform(vectorized_text)
    return scaled_text

# Define a function to make predictions
def predict(input_text):
    preprocessed_input = preprocess_input(input_text)
    prediction = grid.predict(preprocessed_input)[0]  # Assuming only one prediction per input
    return prediction

# Function to read uploaded file
def read_file(file):
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext == '.csv':
        df = pd.read_csv(file, encoding='latin1')
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file)
    else:
        df = pd.DataFrame()
    return df

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    message = None
    predictions = None
    output_filename = None
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        df = read_file(file)
        if df.empty:
            message = "Invalid file format. Please upload a CSV or Excel file."
        else:
            predictions = [predict(text) for text in df.iloc[:, 0]]  # Assuming the relevant text is in the first column
            df['Prediction'] = predictions
            output_filename = 'predictions.xlsx'
            df.to_excel(output_filename, index=False)
    elif 'text' in request.form and request.form['text'] != '':
        text = request.form['text']
        prediction = predict(text)
        predictions = [prediction]
        df = pd.DataFrame({'Requirement': [text], 'Prediction': predictions})
        # output_filename = 'prediction.xlsx'
        # df.to_excel(output_filename, index=False)
    else:
        message = "Please upload a file or enter text."
    
    return render_template('index.html', predictions=predictions, message=message, output_filename=output_filename)

# Route for downloading the predictions file
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host= '0.0.0.0', port=5000)