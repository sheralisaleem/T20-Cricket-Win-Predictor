from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Load the trained Keras model and the StandardScaler
# Ensure these files are in the same directory as app.py
try:
    model = tf.keras.models.load_model('model_ann.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    target = int(data['target'])
    current_score = int(data['current_score'])
    balls_left = int(data['balls_left'])
    wickets_left = int(data['wickets_left'])
    
    # Calculate CRR (Current Run Rate)
    overs_bowled = (120 - balls_left) / 6
    crr = (current_score / overs_bowled) if overs_bowled > 0 else 0
    
    # Calculate RRR (Required Run Rate)
    runs_left = target - current_score
    overs_left = balls_left / 6
    rrr = (runs_left / overs_left) if overs_left > 0 else 0
    
    # Stop prediction if the match is already over
    if current_score >= target:
        return jsonify({'win_prob': 100.0, 'loss_prob': 0.0, 'message': 'Chasing Team Won!'})
    elif balls_left <= 0 or wickets_left <= 0:
        return jsonify({'win_prob': 0.0, 'loss_prob': 100.0, 'message': 'Defending Team Won!'})

    # Prepare features in the exact order the model was trained on
    # [target, current_score, balls_left, wickets_left, crr, rrr]
    features = np.array([[target, current_score, balls_left, wickets_left, crr, rrr]])
    
    # Scale features using the fitted scaler
    features_scaled = scaler.transform(features)
    
    # Get Win Probability for the Chasing Team from the Keras ANN
    win_prob = model.predict(features_scaled, verbose=0)[0][0]
    win_prob_percent = round(float(win_prob * 100), 2)
    loss_prob_percent = round(100 - win_prob_percent, 2)
    
    return jsonify({
        'win_prob': win_prob_percent,
        'loss_prob': loss_prob_percent,
        'crr': round(crr, 2),
        'rrr': round(rrr, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)