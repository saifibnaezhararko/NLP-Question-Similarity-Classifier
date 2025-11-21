# README.md for Quora Question Pairs Project

```markdown
# 🔍 Quora Question Pairs Duplicate Detection

A deep learning project that identifies duplicate question pairs using advanced NLP techniques and a Siamese Neural Network architecture. **Achieved 80.84% accuracy and 90.46% AUC-ROC** on 82K+ test samples.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20.0](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎉 Quick Results

| Metric | Value | Details |
|--------|-------|---------|
| 🎯 **Accuracy** | **80.84%** | On 82,284 test pairs |
| 📊 **AUC-ROC** | **90.46%** | Excellent discrimination |
| 🔍 **Precision** | **69.17%** | Not Duplicate: 92% |
| 🎪 **Recall** | **89.67%** | Catches most duplicates |
| ⚡ **Training Time** | **~17 min** | 5 epochs on CPU |
| 🧠 **Parameters** | **4.7M** | 17.96 MB model size |

**Best Performing Epoch:** 3 (Validation AUC: 90.73%)

## 📋 Table of Contents

- [Quick Results](#-quick-results)
- [Quick Start](#-quick-start)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Training History](#-training-history)
- [Examples](#examples)
- [System Information](#-system-information)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/saifibnaezhararko/NLP-Question-Similarity-Classifier
cd NLP-Question-Similarity-Classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# Place train.csv in the project root

# 4. Train the model
python quora_duplicate_detection.py

# Expected: ~17 minutes training, 80.84% accuracy
```

## 🎯 Overview

This project implements a sophisticated duplicate question detection system using:
- **Advanced Text Preprocessing**: NLTK-based tokenization, lemmatization, and stopword removal
- **Feature Engineering**: 16+ handcrafted features including Jaccard similarity, fuzzy matching, Levenshtein distance
- **Deep Learning**: Bidirectional LSTM-based Siamese network with multiple merge strategies
- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations

## ✨ Features

- 🧠 **Advanced Siamese Network** with shared embeddings
- 📊 **Comprehensive EDA** with word clouds, distributions, and correlation analysis
- 🔧 **Rich Feature Engineering**:
  - Text similarity metrics (Jaccard, Cosine, Sequence Matcher)
  - Fuzzy matching scores (4 variants)
  - Levenshtein distance
  - Statistical features (length, word count, differences)
- 📈 **Multiple Merge Strategies**: Concatenation, absolute difference, multiplication, dot product
- 🎨 **Visualization**: Training history, confusion matrix, ROC curves
- 💾 **Model Persistence**: Save/load trained models and preprocessing artifacts

## 📁 Project Structure

```
quora-question-pairs/
│
├── train.csv                          # Training dataset (download separately)
├── quora_duplicate_detection.py       # Main training script
├── README.md                          # This file
│
├── Models/ (generated after training)
│   ├── quora_advanced_model.keras     # Trained model
│   ├── best_model.keras               # Best checkpoint
│   ├── tokenizer.pkl                  # Text tokenizer
│   ├── scaler.pkl                     # Feature scaler
│   └── config.pkl                     # Model configuration
│
└── requirements.txt                   # Python dependencies
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, but recommended for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/saifibnaezhararko/NLP-Question-Similarity-Classifier
cd quora-question-pairs
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0
scikit-learn>=0.24.0
tensorflow>=2.20.0
wordcloud>=1.8.0
```

### Step 4: Download NLTK Data

The script automatically downloads required NLTK data, but you can manually download:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

## 📊 Dataset

### Download Dataset

Download the Quora Question Pairs dataset from [Kaggle](https://www.kaggle.com/datasets/sksaifibnaezhararko/train-csv):

```bash
# Using Kaggle API
kaggle competitions download -c quora-question-pairs

# Or download manually and place train.csv in project root
```

### Dataset Structure

The `train.csv` file should contain:
- `id`: Question pair ID
- `qid1`, `qid2`: Unique question IDs
- `question1`: First question text
- `question2`: Second question text
- `is_duplicate`: Target variable (0 or 1)

### Dataset Statistics

- **Total Pairs**: 404,351 (after preprocessing: 403,956)
- **Duplicates**: 36.92% (149,306 pairs)
- **Non-duplicates**: 63.08% (255,045 pairs)
- **Training Set**: 329,135 samples
- **Test Set**: 82,284 samples
- **Augmented Samples**: 7,463 (final dataset: 411,419)

## 🚀 Usage

### Training the Model

```bash
python quora_duplicate_detection.py
```

The script will:
1. ✅ Load and preprocess data (404,351 pairs → 403,956 after cleanup)
2. 📊 Perform exploratory data analysis
3. 🔧 Engineer 16 advanced features
4. 🧠 Build and train Siamese network (4.7M parameters)
5. 📈 Evaluate performance (80.84% accuracy, 90.46% AUC)
6. 💾 Save model artifacts (model, tokenizer, scaler, config)

**Expected Output:**
```
✓ All libraries imported successfully!
Dataset shape: (404351, 6)
After preprocessing: (403956, 8)
✓ Feature engineering completed! (16 features)
Training advanced model...
Epoch 1/5 - val_accuracy: 0.8002 - val_auc: 0.8983
...
Best Model: Epoch 3 - val_auc: 0.9073
✓ Model saved successfully!
```

### Training Parameters

Configuration used in the successful training run:

```python
MAX_VOCAB_SIZE = 30000           # Vocabulary size
MAX_SEQUENCE_LENGTH = 15         # Max words per question
EMBEDDING_DIM = 128              # Embedding dimension
LSTM_UNITS = 128                 # LSTM units
BATCH_SIZE = 128                 # Training batch size
EPOCHS = 5                       # Training epochs
```

### Making Predictions

After training, the following artifacts are saved:
- `quora_advanced_model.keras` - Final trained model
- `best_model.keras` - Best checkpoint (Epoch 3, Val AUC: 0.9073)
- `tokenizer.pkl` - Text tokenizer for preprocessing
- `scaler.pkl` - Feature scaler for numerical features
- `config.pkl` - Model configuration

```python
import pickle
from tensorflow import keras

# Load model and artifacts
model = keras.models.load_model('quora_advanced_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict
question1 = "How do I learn Python?"
question2 = "What's the best way to learn Python programming?"

prediction, probability = predict_similarity(
    question1, question2, model, tokenizer, scaler
)

print(f"Duplicate: {prediction}")
print(f"Confidence: {probability:.2%}")
```

## 🌟 What Makes This Implementation Special?

1. **Hybrid Approach**: Combines deep learning (LSTM) with traditional ML features (Jaccard, fuzzy matching)
2. **Multiple Merge Strategies**: Uses 4 different ways to compare question embeddings
3. **Data Augmentation**: Automatically generates 7,463 synthetic samples for better generalization
4. **Production-Ready**: Includes model persistence, error handling, and comprehensive logging
5. **Class Balancing**: Uses class weights (0.808, 1.313) to handle imbalanced data (63% vs 37%)
6. **Early Stopping**: Automatically saves best model and prevents overfitting
7. **Comprehensive Evaluation**: Provides confusion matrix, ROC curves, and feature importance

**Real-World Performance:**
- ✅ Successfully detects paraphrased questions (93.92% confidence)
- ✅ Correctly rejects unrelated questions (1.37% confidence)
- ⚠️ Moderate performance on structurally similar but semantically different questions

## 🏗️ Model Architecture

### Siamese Network Design

```
Input Layer (Question 1)          Input Layer (Question 2)
        ↓                                    ↓
   Embedding (Shared)                 Embedding (Shared)
        ↓                                    ↓
Bidirectional LSTM (Shared)      Bidirectional LSTM (Shared)
        ↓                                    ↓
        └────────────── Merge ───────────────┘
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                  ↓
   Concatenate      Absolute Diff      Multiplication
        ↓                 ↓                  ↓
        └─────────────────┴──────────────────┘
                          ↓
              Numerical Features (16)
                          ↓
                Dense (512) + BN + Dropout
                          ↓
                Dense (256) + BN + Dropout
                          ↓
                Dense (128) + BN + Dropout
                          ↓
                  Sigmoid Output
```

### Key Components

- **Shared Embeddings**: Both questions use the same embedding layer
- **Bidirectional LSTM**: Captures context from both directions
- **Multiple Merge Strategies**: 
  - Concatenation
  - Absolute difference
  - Element-wise multiplication
  - Dot product (cosine similarity)
- **Batch Normalization**: Stabilizes training
- **Dropout Layers**: Prevents overfitting

## 📊 Results

### Model Performance

| Metric    | Score  | Details |
|-----------|--------|---------|
| Accuracy  | 80.84% | Overall correctness |
| Precision | 0.6917 | 69% of predicted duplicates are correct |
| Recall    | 0.8967 | 90% of actual duplicates detected |
| F1-Score  | 0.7810 | Harmonic mean of precision & recall |
| AUC-ROC   | 0.9046 | Excellent discrimination ability |

### Detailed Classification Report

```
                precision    recall  f1-score   support

Not Duplicate       0.92      0.75      0.83     50,940
    Duplicate       0.69      0.90      0.78     31,344

     accuracy                           0.81     82,284
    macro avg       0.81      0.83      0.81     82,284
 weighted avg       0.83      0.81      0.81     82,284
```

**Key Insights:**
- **High Recall (90%)**: Great at catching duplicate questions
- **Moderate Precision (69%)**: Some false positives (predicts duplicate when not)
- **Excellent for Not Duplicates**: 92% precision, 75% recall

### Training Visualizations

The script generates:
- 📉 Loss and accuracy curves
- 📊 Confusion matrix
- 📈 ROC curve
- ☁️ Word clouds for duplicate/non-duplicate questions
- 📊 Feature correlation heatmap

### Confusion Matrix

```
                    Predicted
                Not Dup    Duplicate
Actual  Not Dup   38,205    12,735  (Precision: 0.92)
        Duplicate  3,234     28,110  (Recall: 0.90)

Overall Accuracy: 80.84%
```

## 💡 Examples

### Duplicate Pairs (High Similarity)

```
✓ Q1: "How do I learn Python?"
  Q2: "What's the best way to learn Python programming?"
  → Duplicate (Confidence: 93.92%) ✅

✓ Q1: "How to reverse a string in Python?"
  Q2: "How to reverse a string in Python programming?"
  → Duplicate (Confidence: 88.86%) ✅

⚠️ Q1: "How can I lose weight?"
   Q2: "What are effective weight loss strategies?"
   → Duplicate (Confidence: 59.83%) - Review recommended
```

### Non-Duplicate Pairs (Low Similarity)

```
✓ Q1: "What is machine learning?"
  Q2: "How do I bake a cake?"
  → Not Duplicate (Confidence: 1.37%) ✅

❌ Q1: "What is the capital of France?"
   Q2: "What is the capital of Germany?"
   → Predicted as Duplicate (Confidence: 54.18%) ❌
   → Model limitation: Structurally similar questions
```

**Note:** Confidence represents the model's certainty that questions are duplicates. Lower confidence (below 50%) indicates non-duplicates.

## 📊 Training History

### Epoch Performance

| Epoch | Train Acc | Train AUC | Val Acc | Val AUC | Val Loss |
|-------|-----------|-----------|---------|---------|----------|
| 1     | 0.7633    | 0.8578    | 0.8002  | 0.8983  | 0.4051   |
| 2     | 0.8173    | 0.9070    | 0.8080  | 0.9066  | 0.3985   |
| 3     | 0.8459    | 0.9288    | 0.8103  | **0.9073** | 0.4083   |
| 4     | 0.8679    | 0.9441    | 0.8148  | 0.9053  | 0.4187   |
| 5     | 0.8842    | 0.9552    | 0.8161  | 0.9024  | 0.4368   |

**Best Model:** Epoch 3 with validation AUC of 0.9073
**Early Stopping:** Triggered after Epoch 5 (patience: 2)

### Key Observations

✅ **Strengths:**
- Excellent at detecting semantically similar questions (93.92% confidence for Python learning questions)
- High recall (89.67%) - catches most duplicate pairs
- Strong AUC-ROC (90.46%) - good discrimination ability

⚠️ **Limitations:**
- Can struggle with structurally similar but semantically different questions
- Example: "Capital of France" vs "Capital of Germany" predicted as duplicate (54.18% confidence)
- Moderate precision (69.17%) - some false positives

**Recommendations:**
- Use confidence threshold of 70%+ for production
- Review predictions between 50-70% confidence manually
- Consider adding semantic understanding features (word embeddings comparison)

## 📊 Feature Statistics

From the feature engineering analysis (403,956 question pairs):

| Feature | Mean | Median | Std Dev |
|---------|------|--------|---------|
| Jaccard Similarity | 0.415 | 0.375 | 0.280 |
| Word Share | 0.265 | 0.273 | 0.143 |
| Sequence Similarity | 0.599 | 0.600 | 0.230 |
| Fuzz Ratio | 0.613 | 0.600 | 0.186 |
| Cosine Similarity | 0.545 | 0.577 | 0.286 |
| Levenshtein (normalized) | 0.492 | 0.448 | 0.247 |

**Data Augmentation:** Added 7,463 synthetic samples for better generalization

## 💻 System Information

- **TensorFlow Version:** 2.20.0
- **oneDNN:** Enabled (custom operations)
- **Platform:** Windows with Python 3.12
- **Total Parameters:** 4,706,817 (17.96 MB)
- **Trainable Parameters:** 4,705,025 (17.95 MB)
- **Training Time:** ~17 minutes (5 epochs)
- **Average Epoch Time:** ~3.5 minutes

## 🔬 Advanced Features

### Feature Engineering (16 Features)

1. **Basic Features**:
   - Question lengths (characters)
   - Word counts
   - Length differences
   - Common words count

2. **Similarity Metrics**:
   - Jaccard similarity
   - Word share ratio
   - Sequence similarity
   - Cosine similarity

3. **Fuzzy Matching**:
   - Simple ratio
   - Partial ratio
   - Token sort ratio
   - Token set ratio

4. **Edit Distance**:
   - Normalized Levenshtein distance

### Text Preprocessing Pipeline

1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Special character removal
5. Tokenization
6. Stopword removal
7. Lemmatization

## ⚙️ Configuration

### GPU Configuration

For faster training with GPU:

```python
import tensorflow as tf

# Check GPU availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Hyperparameter Tuning

Modify these parameters in the script:

```python
# Model parameters (Actual Configuration)
MAX_VOCAB_SIZE = 30000          # Vocabulary size
MAX_SEQUENCE_LENGTH = 15        # Max words per question
EMBEDDING_DIM = 128            # Embedding dimension
LSTM_UNITS = 128               # LSTM units (Bidirectional = 256)
dropout_rate = 0.4             # Dropout rate

# Training parameters (Used in Training)
batch_size = 128               # Batch size
learning_rate = 0.0003         # Learning rate (Adam optimizer)
epochs = 5                     # Maximum epochs
early_stopping_patience = 2    # Early stopping patience
class_weights = {0: 0.808, 1: 1.313}  # For imbalanced data

# Training Results
# Best validation AUC: 0.9073 at Epoch 3
# Final test accuracy: 80.84%
# Total training time: ~17 minutes
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**Issue**: TensorFlow oneDNN warnings
```bash
# These are informational, not errors. To suppress:
set TF_ENABLE_ONEDNN_OPTS=0  # Windows
export TF_ENABLE_ONEDNN_OPTS=0  # Linux/Mac
```

**Issue**: Out of Memory Error
```python
# Solution: Reduce batch_size or MAX_SEQUENCE_LENGTH
BATCH_SIZE = 64  # instead of 128
MAX_SEQUENCE_LENGTH = 10  # instead of 15
```

**Issue**: NLTK data not found
```python
# Solution: Download manually
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

**Issue**: Slow training on CPU
```
Solution: 
- Expected: ~3.5 minutes per epoch on CPU
- Use GPU for 3-5x speedup
- Reduce dataset with sampling (currently at 100%)
- Reduce LSTM_UNITS from 128 to 64
```

**Issue**: Model predictions seem inconsistent
```
Solution:
- Set confidence threshold to 70%+ for production use
- Manually review predictions between 50-70% confidence
- Model can struggle with structurally similar questions
  (e.g., "Capital of X" vs "Capital of Y")
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- [ ] Add transformer-based models (BERT, RoBERTa)
- [ ] Implement cross-validation
- [ ] Add hyperparameter optimization
- [ ] Create web API for predictions
- [ ] Add more data augmentation techniques
- [ ] Implement attention mechanisms

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Quora Question Pairs](https://www.kaggle.com/datasets/sksaifibnaezhararko/train-csv)
- **Libraries**: TensorFlow, scikit-learn, NLTK, and all amazing open-source contributors

## 📧 Contact

Sk Saif Ibna Ezhar Arko - [saifibnaezhararko1@gmail.com](https://saifibnaezhararko.github.io/)

Project Link: [https://github.com/saifibnaezhararko/NLP-Question-Similarity-Classifier](https://github.com/saifibnaezhararko/NLP-Question-Similarity-Classifier)

---

⭐ **Star this repo if you find it helpful!** ⭐

---
## 📈 Project Status & Roadmap

### ✅ Completed Features (v1.0)
- ✅ **Siamese LSTM Network** (4.7M parameters, 80.84% accuracy)
- ✅ **Advanced Feature Engineering** (16 features)
- ✅ **Comprehensive EDA** with visualizations
- ✅ **Data Augmentation** (7,463 synthetic samples)
- ✅ **Model Persistence** (save/load functionality)
- ✅ **Batch Normalization & Dropout** for regularization
- ✅ **Early Stopping** with model checkpointing
- ✅ **Class Weighting** for imbalanced data
- ✅ **Multiple Merge Strategies** (concat, diff, multiply, dot)

### 🔄 Planned Enhancements (v2.0)
- 🔄 **Transformer Integration** (BERT/RoBERTa for better semantic understanding)
- 🔄 **FastAPI Web Service** (REST API for predictions)
- 🔄 **Docker Containerization** (easy deployment)
- 🔄 **Model Monitoring Dashboard** (track performance)
- 🔄 **A/B Testing Framework** (compare model versions)
- 🔄 **Improved Precision** (reduce false positives)

**Current Version:** v1.0 (Trained on 411K pairs, validated on 82K)
```