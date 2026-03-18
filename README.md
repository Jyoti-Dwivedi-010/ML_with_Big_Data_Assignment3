# Recommender Systems Assignment

## CSL7110 - Machine Learning with Big Data

### Overview

This Jupyter notebook implements various recommender system algorithms as part of the ML with Big Data course assignment.

### Dataset

- **MovieLens ml-latest-small** dataset
- Contains 100,000+ ratings from 600+ users on 9,000+ movies
- Files used: `movies.csv`, `ratings.csv`, `tags.csv`

### Contents

#### Part 1: Content-Based Filtering (20 marks)

- **Task 1**: TF-IDF Based Recommendation
    - Genre extraction and TF-IDF vectorization
    - Cosine similarity computation
    - Movie recommendation function with sample queries
- **Task 2**: User-Profile-Based Content Recommender
    - Weighted user profile construction
    - User-movie similarity computation
    - Precision@K and Recall@K evaluation

#### Part 2: Collaborative Filtering (20 marks)

- **Task 3**: User-Based Collaborative Filtering
    - User-movie rating matrix construction
    - Pearson correlation similarity
    - Weighted k-NN prediction
    - RMSE, Precision@K, Recall@K evaluation
- **Task 4**: Item-Based Collaborative Filtering
    - Item-item similarity computation
    - Rating prediction and recommendation
    - Comparison with user-based CF

#### Part 3: Matrix Factorization (20 marks)

- **Task 5**: SVD Implementation
    - Manual SVD using scipy
    - Rating matrix reconstruction
    - Missing rating prediction
- **Task 6**: Surprise Library SVD
    - Grid search hyperparameter tuning
    - Performance comparison with manual implementation

#### Part 4: Hybrid Model (10 marks)

- **Task 7**: Meta-Learning Hybrid Recommender
    - Combining CBF and CF scores
    - Random Forest/Gradient Boosting meta-model
    - Cold-start user analysis

#### Part 5: Learning-Based Systems (40 marks)

- **Task 8**: Neural Network Content-Based Filtering
    - User and movie embeddings
    - TensorFlow/Keras implementation
    - Training and evaluation
- **Task 9**: Reinforcement Learning Recommender
    - ε-Greedy Multi-Armed Bandit
    - UCB (Upper Confidence Bound) Bandit
    - Q-Learning agent
    - Comparison with traditional methods

#### Part 6: Explainability (10 marks)

- **Task 10**: SHAP explanations for feature importance
- **Task 11**: Neighborhood-based explanations for CF
- **Task 12**: LIME for neural network interpretability
- **Task 13**: Comparative evaluation of explainability methods

### Requirements

#### Python Version

- Python 3.8+

#### Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy scikit-surprise tensorflow shap lime jupyter
```

### Running the Notebook

1. **Setup Environment**:

    ```bash
    cd /path/to/MLBD
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. **Extract Dataset** (if not already extracted):

    ```bash
    unzip ml-latest-small.zip
    ```

3. **Launch Jupyter**:

    ```bash
    jupyter notebook Recommender_Systems_Assignment.ipynb
    ```

4. **Run All Cells**:
    - Select `Cell > Run All` from the menu
    - Or run cells individually using Shift+Enter

### Expected Output

- Recommendation results for sample queries
- Performance metrics (RMSE, Precision@K, Recall@K)
- Comparison tables across methods
- Visualizations (learning curves, heatmaps, bar charts)
- Explainability analyses

### File Structure

```
MLBD/
├── Recommender_Systems_Assignment.ipynb  # Main notebook
├── README.md                              # This file
├── images folder (output images)
├── requirements.txt
├── M25CSA010_CSL7110_Assignment3 (report)
├── CSL7110 Assignment 3 -ML with Big Data( assignment)
├── ml-latest-small/                       # Dataset folder
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── links.csv
│   └── README.txt
└── .ipynb_checkpoints
```

### Notes

- The notebook uses random seeds for reproducibility
- Some cells may take several minutes to run (especially neural network training and RL)
- All code cells are properly commented and documented
- Visualizations are included throughout for better understanding

### Author

Assignment for CSL7110 - ML with Big Data Course
