# T20 Match Predictor - AI Win Probability Forecaster

A machine learning project predicting chasing team win probability in T20 cricket using real-time match metrics and a neural network.

## Demo

![T20 Match Predictor Demo](https://github.com/sheralisaleem/T20-Cricket-Win-Predictor/releases/download/v1.0/demo.gif)

## Features

✅ **Deep Learning Model** - ANN with 2 hidden layers predicting win probabilities  
✅ **Feature Engineering** - Calculates CRR, RRR, wickets remaining, balls remaining  
✅ **Interactive Web Interface** - Flask + Bootstrap UI for live match simulation  
✅ **Model Evaluation** - Classification metrics, confusion matrix, ROC-AUC scoring

## Project Structure

```
t20_match_predictor/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.ipynb                    # Main Jupyter notebook with full pipeline
├── app.py                       # Flask backend server
├── model_ann.keras              # Trained neural network model
├── scaler.pkl                   # StandardScaler for feature normalization
├── templates/
│   └── index.html              # Interactive web interface
├── ball_by_ball_it20.csv       # T20 cricket ball-by-ball dataset (excluded from git)
└── venv/                        # Python virtual environment
```

## Installation

1. **Python 3.8+ required**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download dataset** from [Kaggle Ball-by-Ball IT20](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-it20/data)
4. **Extract as** `ball_by_ball_it20.csv` in project directory

## Usage

### 1. Train Model (Jupyter Notebook)
```bash
jupyter notebook app.ipynb
```
Cells: Load data → Explore → Engineer features → Split data → Train ANN → Evaluate → Generate `model_ann.keras` & `scaler.pkl`

### 2. Live Web Interface (Flask)
```bash
python app.py
# Open http://localhost:5000
```
- Enter target, initial score, wickets, balls left
- Click buttons to simulate runs/wickets/extras
- Watch win probability update in real-time
- Dark/light theme toggle available

## Frontend

Flask + Bootstrap web interface for real-time match simulation:
- Interactive buttons for runs (0-6), wickets, extras
- Live probability bar (chasing vs defending)
- Dark/light theme toggle
- Undo/Restart controls
- Real-time CRR, RRR, and win probability calculations

## Data Format

CSV columns: Match ID, Innings, Over, Runs From Ball, Innings Runs, Innings Wickets, Target Score, Balls Remaining, Chased Successfully

## Model Architecture

Input (6 features) → Dense(16, relu) → Dense(8, relu) → Output(1, sigmoid)

**Training**: Adam optimizer, binary crossentropy, 20 epochs, batch size 32, 80/20 split

## Workflow

Load Data → Explore → Engineer Features → Clean → Split → Scale → Train → Evaluate → Predict

## Notes

- Educational project demonstrating ML pipeline in cricket analytics
- Model performance depends on data quality and feature distribution
- Consider adding: bowler/batter stats, team strength, weather, venue factors

## Author

**Sher Ali Saleem**

Created as a T20 cricket analytics and machine learning learning project.

---

**Last Updated**: May 2026  
**Data Source**: International T20 ball-by-ball records
