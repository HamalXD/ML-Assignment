from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model_filename = 'models/best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    year_start = int(request.form['YearStart'])
    location_id = request.form['LocationID']
    class_id = request.form['ClassID']
    topic_id = request.form['TopicID']
    question_id = request.form['QuestionID']

    # Prepare the input data as per the model's training
    input_data = np.array([[year_start, location_id, class_id, topic_id, question_id]])
    input_data_encoded = np.zeros(model.n_features_in_)  # Placeholder for encoded features
    input_data_encoded[0] = year_start  # YearStart is assumed to be the first feature

    # Predict using the loaded model
    prediction = model.predict([input_data_encoded])[0]
    prediction_label = 'Above Median' if prediction == 1 else 'Below or Equal to Median'

    # Render the prediction result on the page
    return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
