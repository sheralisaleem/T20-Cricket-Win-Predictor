# T20 Match Predictor - AI Win Probability Forecaster

A machine learning project that predicts the probability of a chasing team winning in International T20 cricket matches using real-time match metrics and deep learning.

## Project Overview

This project analyzes ball-by-ball T20 cricket data to build an **Artificial Neural Network (ANN)** that predicts win probabilities for chasing teams. The model considers dynamic match factors like:
- Current run rate (CRR)
- Required run rate (RRR)
- Remaining wickets
- Target score
- Current score
- Balls remaining

## Features

✅ **Data Loading & EDA**: Comprehensive exploratory data analysis with visualizations
- Win/loss distribution pie charts
- Target score distributions
- Average runs per over analysis

✅ **Feature Engineering**: Automated calculation of cricket-specific metrics
- Current Run Rate (CRR)
- Required Run Rate (RRR)
- Wickets remaining
- Balls remaining

✅ **Deep Learning Model**: 
- Sequential neural network with 2 hidden layers (16 and 8 neurons)
- ReLU activation for hidden layers, Sigmoid for output
- Binary classification with Keras/TensorFlow

✅ **Model Evaluation**:
- Classification reports with precision, recall, F1-score
- Confusion matrix visualization
- ROC-AUC scoring
- Live match prediction simulation

## Project Structure

```
t20_match_predictor/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.ipynb                    # Main Jupyter notebook with full pipeline
└── ball_by_ball_it20.csv       # T20 cricket ball-by-ball dataset
```

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or VS Code with Jupyter extension

### Required Libraries

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy           # Data manipulation
pip install scikit-learn          # Machine learning utilities
pip install tensorflow            # Neural network framework
pip install matplotlib seaborn    # Visualization
```

## Dataset Setup

The project requires the T20 cricket ball-by-ball dataset. Download it from Kaggle:

1. Visit: [Ball-by-Ball IT20 Dataset](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-it20/data)
2. Click **Download** button
3. Extract the CSV file to the project directory
4. Rename it to `ball_by_ball_it20.csv` (or update the filename in the notebook)

**Note**: The CSV file is excluded from version control by `.gitignore`.

## Usage

### Running the Notebook

1. Open the notebook in Jupyter or VS Code:
   ```bash
   jupyter notebook app.ipynb
   ```
   Or in VS Code: Open `app.ipynb` directly

2. Execute cells sequentially from top to bottom:
   - **Cell 1-3**: Load and explore data
   - **Cell 4-5**: Feature engineering
   - **Cell 6-7**: Data preparation and splitting
   - **Cell 8**: Train the neural network model
   - **Cell 9**: Live prediction simulation
   - **Cell 10**: Advanced evaluation metrics and visualizations

### Making a Live Prediction

In the "Live Match Prediction" section, modify these variables:
```python
target_score = 185           # Target the chasing team needs
current_score = 120          # Current score of chasing team
balls_left = 30              # Balls remaining in the match
wickets_left = 6             # Wickets remaining (10 - wickets_lost)
```

The model will output the win probability percentage for the chasing team.

## Data Format

The `ball_by_ball_it20.csv` should contain columns like:
- `Match ID`: Unique match identifier
- `Innings`: 1 for batting first, 2 for chasing
- `Over`: Over number (1-20)
- `Runs From Ball`: Runs scored on that delivery
- `Innings Runs`: Cumulative runs in that innings
- `Innings Wickets`: Cumulative wickets lost
- `Target Score`: Target for chasing team
- `Balls Remaining`: Balls left in the innings
- `Chased Successfully`: Binary target (1 = Win, 0 = Loss)

## Model Architecture

```
Input Layer (6 features)
    ↓
Dense(16, activation='relu')   # Hidden Layer 1
    ↓
Dense(8, activation='relu')    # Hidden Layer 2
    ↓
Dense(1, activation='sigmoid') # Output Layer (probability)
```

**Compiler**: Adam optimizer with binary crossentropy loss
**Training**: 20 epochs, batch size 32, 20% validation split

## Key Results

- **Accuracy**: Model trained on 80% of data
- **Evaluation**: Tested on hidden 20% test set
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrix and ROC curve plots

## Technologies Used

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical calculations (CRR, RRR) |
| **scikit-learn** | Data splitting, scaling, metrics |
| **TensorFlow/Keras** | Neural network model |
| **matplotlib** | Static visualizations |
| **seaborn** | Statistical plots and heatmaps |

## Workflow

1. **Load Data** → Read CSV into pandas DataFrame
2. **Explore** → Analyze distributions and statistics
3. **Engineer** → Calculate CRR, RRR, wickets remaining
4. **Clean** → Remove missing values and invalid rows
5. **Split** → 80/20 train-test split
6. **Scale** → StandardScaler normalization
7. **Train** → Fit ANN on training data
8. **Evaluate** → Classification report, confusion matrix, ROC curve
9. **Predict** → Use live scenario with scaled data

## Tips

- Always scale live match data using the **same scaler** trained on training data
- Ensure no missing values in feature columns
- Check that `balls_left` and `wickets_left` are non-negative
- Validate that data distributions match training data before deployment

## Output Examples

```
Training data size: 45000 balls
Testing data size: 11250 balls

Deep Learning Accuracy: 78.45%

--- LIVE MATCH PREDICTION (NEURAL NETWORK) ---
Scenario: Chasing 185, Currently at 120/4
Required Run Rate: 14.20
Win Probability for Chasing Team: 62.3%
```

## Notes

- This is an educational project demonstrating ML/DL pipeline in cricket analytics
- Model performance depends heavily on data quality and feature distribution
- Real-world deployment would require more sophisticated error handling and validation
- Consider adding more features like:
  - Bowler/Batter statistics
  - Team strength ratings
  - Weather conditions
  - Venue factors

## Author

**Sher Ali Saleem**

Created as a T20 cricket analytics and machine learning learning project.

---

**Last Updated**: May 2026  
**Data Source**: International T20 ball-by-ball records
