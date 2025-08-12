from flask import Flask, request, render_template
import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('Vector.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Climate.html')  # HTML template for input form

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text']
    if not text_input.strip():
        return render_template('Climate.html', prediction="Please enter some text.")
    
    # Preprocess and vectorize
    clean_text = text_input.lower()
    vectorized_text = vectorizer.transform([clean_text])
    
    # Dummy engagement (0s)
    engagement = np.array([[0, 0]])
    final_input = np.hstack([vectorized_text.toarray(), engagement])
    
    # Predict
    prediction = model.predict(final_input)[0]
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}  
    prediction_label = label_map[prediction]

    return render_template('Climate.html', prediction=f"The sentiment is: {prediction_label.capitalize()}")

if __name__ == '__main__':
    app.run(debug=True)
