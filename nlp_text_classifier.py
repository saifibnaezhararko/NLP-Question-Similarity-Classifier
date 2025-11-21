# Install additional required packages
#!pip install python-Levenshtein fuzzywuzzy python-Levenshtein-wheels -q

#pip install tensorflow

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text processing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

# Advanced text similarity
from fuzzywuzzy import fuzz
import Levenshtein

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Embedding, Dropout,
    Bidirectional, concatenate, BatchNormalization, Lambda,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Add
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("✓ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

"""## 2. Load Data"""

# Load the training data
df = pd.read_csv('train.csv', engine='python')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df = df.dropna(subset=['is_duplicate'])

print(f"\nAfter handling missing values: {df.shape}")

# Target distribution
print("\nTarget Distribution:")
print(df['is_duplicate'].value_counts())
print("\nPercentage:")
print(df['is_duplicate'].value_counts(normalize=True) * 100)

"""## 2.5. Comprehensive Exploratory Data Analysis (EDA)

### Step 1: Understanding the Data Distribution and Patterns

This section performs detailed exploratory analysis as required:
- Visualize target variable distribution
- Analyze text characteristics (length, word count, character count)
- Word clouds for duplicate vs non-duplicate questions
- Common words analysis
- Correlation between features
"""

# Install wordcloud if needed
try:
    from wordcloud import WordCloud
except:
    import subprocess
    subprocess.run(['pip', 'install', 'wordcloud', '-q'])
    from wordcloud import WordCloud

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Create a sample for faster EDA (if dataset is large)
eda_df = df.copy()
print(f"Analyzing {len(eda_df):,} question pairs...")

### 2.5.1 Target Variable Distribution
print("\n1. Target Variable Distribution:")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
eda_df['is_duplicate'].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('Distribution of Duplicate vs Non-Duplicate Questions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Is Duplicate', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Not Duplicate (0)', 'Duplicate (1)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, v in enumerate(eda_df['is_duplicate'].value_counts().values):
    axes[0].text(i, v + 1000, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)
eda_df['is_duplicate'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                            colors=colors, explode=explode, startangle=90)
axes[1].set_title('Proportion of Duplicate Questions', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

### 2.5.2 Text Length Analysis
print("\n2. Text Length Analysis:")
print("-" * 60)

# Calculate lengths
eda_df['q1_length'] = eda_df['question1'].astype(str).apply(len)
eda_df['q2_length'] = eda_df['question2'].astype(str).apply(len)
eda_df['q1_words'] = eda_df['question1'].astype(str).apply(lambda x: len(x.split()))
eda_df['q2_words'] = eda_df['question2'].astype(str).apply(lambda x: len(x.split()))

print(f"\nCharacter Length Statistics:")
print(f"Question 1 - Mean: {eda_df['q1_length'].mean():.1f}, Median: {eda_df['q1_length'].median():.1f}")
print(f"Question 2 - Mean: {eda_df['q2_length'].mean():.1f}, Median: {eda_df['q2_length'].median():.1f}")

print(f"\nWord Count Statistics:")
print(f"Question 1 - Mean: {eda_df['q1_words'].mean():.1f}, Median: {eda_df['q1_words'].median():.1f}")
print(f"Question 2 - Mean: {eda_df['q2_words'].mean():.1f}, Median: {eda_df['q2_words'].median():.1f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Character length distribution
axes[0, 0].hist([eda_df[eda_df['is_duplicate']==0]['q1_length'], 
                 eda_df[eda_df['is_duplicate']==1]['q1_length']], 
                bins=50, alpha=0.7, label=['Not Duplicate', 'Duplicate'], color=['#3498db', '#e74c3c'])
axes[0, 0].set_title('Question 1 - Character Length Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Character Length')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlim(0, 300)

axes[0, 1].hist([eda_df[eda_df['is_duplicate']==0]['q2_length'], 
                 eda_df[eda_df['is_duplicate']==1]['q2_length']], 
                bins=50, alpha=0.7, label=['Not Duplicate', 'Duplicate'], color=['#3498db', '#e74c3c'])
axes[0, 1].set_title('Question 2 - Character Length Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Character Length')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(0, 300)

# Word count distribution
axes[1, 0].hist([eda_df[eda_df['is_duplicate']==0]['q1_words'], 
                 eda_df[eda_df['is_duplicate']==1]['q1_words']], 
                bins=30, alpha=0.7, label=['Not Duplicate', 'Duplicate'], color=['#3498db', '#e74c3c'])
axes[1, 0].set_title('Question 1 - Word Count Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 50)

axes[1, 1].hist([eda_df[eda_df['is_duplicate']==0]['q2_words'], 
                 eda_df[eda_df['is_duplicate']==1]['q2_words']], 
                bins=30, alpha=0.7, label=['Not Duplicate', 'Duplicate'], color=['#3498db', '#e74c3c'])
axes[1, 1].set_title('Question 2 - Word Count Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, 50)

plt.tight_layout()
plt.show()

### 2.5.3 Word Length Comparison Box Plot
print("\n3. Word Length Comparison:")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot for word counts
box_data_words = [
    eda_df[eda_df['is_duplicate']==0]['q1_words'],
    eda_df[eda_df['is_duplicate']==1]['q1_words'],
    eda_df[eda_df['is_duplicate']==0]['q2_words'],
    eda_df[eda_df['is_duplicate']==1]['q2_words']
]
axes[0].boxplot(box_data_words, labels=['Q1 Non-Dup', 'Q1 Dup', 'Q2 Non-Dup', 'Q2 Dup'])
axes[0].set_title('Word Count Comparison by Duplicate Status', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Word Count')
axes[0].grid(alpha=0.3)

# Box plot for character counts
box_data_chars = [
    eda_df[eda_df['is_duplicate']==0]['q1_length'],
    eda_df[eda_df['is_duplicate']==1]['q1_length'],
    eda_df[eda_df['is_duplicate']==0]['q2_length'],
    eda_df[eda_df['is_duplicate']==1]['q2_length']
]
axes[1].boxplot(box_data_chars, labels=['Q1 Non-Dup', 'Q1 Dup', 'Q2 Non-Dup', 'Q2 Dup'])
axes[1].set_title('Character Length Comparison by Duplicate Status', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Character Count')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

### 2.5.4 Word Clouds
print("\n4. Word Clouds:")
print("-" * 60)

# Combine all questions for each category
duplicate_text = ' '.join(eda_df[eda_df['is_duplicate']==1]['question1'].astype(str).tolist() + 
                          eda_df[eda_df['is_duplicate']==1]['question2'].astype(str).tolist())
non_duplicate_text = ' '.join(eda_df[eda_df['is_duplicate']==0]['question1'].astype(str).tolist()[:5000] + 
                              eda_df[eda_df['is_duplicate']==0]['question2'].astype(str).tolist()[:5000])

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Word cloud for duplicate questions
print("Generating word cloud for duplicate questions...")
wordcloud_dup = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Reds', max_words=100).generate(duplicate_text)
axes[0].imshow(wordcloud_dup, interpolation='bilinear')
axes[0].set_title('Word Cloud - Duplicate Questions', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Word cloud for non-duplicate questions
print("Generating word cloud for non-duplicate questions...")
wordcloud_non_dup = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Blues', max_words=100).generate(non_duplicate_text)
axes[1].imshow(wordcloud_non_dup, interpolation='bilinear')
axes[1].set_title('Word Cloud - Non-Duplicate Questions', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

### 2.5.5 Most Common Words Analysis
print("\n5. Most Common Words Analysis:")
print("-" * 60)

from collections import Counter

def get_top_words(text_series, n=20):
    """Get top N most common words"""
    words = ' '.join(text_series.astype(str)).lower().split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
    return Counter(words).most_common(n)

# Get top words for each category
top_words_dup = get_top_words(pd.concat([eda_df[eda_df['is_duplicate']==1]['question1'],
                                         eda_df[eda_df['is_duplicate']==1]['question2']]))
top_words_non_dup = get_top_words(pd.concat([eda_df[eda_df['is_duplicate']==0]['question1'].head(10000),
                                             eda_df[eda_df['is_duplicate']==0]['question2'].head(10000)]))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Duplicate questions top words
words_dup, counts_dup = zip(*top_words_dup)
axes[0].barh(range(len(words_dup)), counts_dup, color='#e74c3c')
axes[0].set_yticks(range(len(words_dup)))
axes[0].set_yticklabels(words_dup)
axes[0].invert_yaxis()
axes[0].set_xlabel('Frequency', fontsize=11)
axes[0].set_title('Top 20 Words in Duplicate Questions', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Non-duplicate questions top words
words_non_dup, counts_non_dup = zip(*top_words_non_dup)
axes[1].barh(range(len(words_non_dup)), counts_non_dup, color='#3498db')
axes[1].set_yticks(range(len(words_non_dup)))
axes[1].set_yticklabels(words_non_dup)
axes[1].invert_yaxis()
axes[1].set_xlabel('Frequency', fontsize=11)
axes[1].set_title('Top 20 Words in Non-Duplicate Questions', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTop 10 words in Duplicate questions:")
for word, count in top_words_dup[:10]:
    print(f"  {word}: {count:,}")

print("\nTop 10 words in Non-Duplicate questions:")
for word, count in top_words_non_dup[:10]:
    print(f"  {word}: {count:,}")

### 2.5.6 Correlation Analysis
print("\n6. Feature Correlation Analysis:")
print("-" * 60)

# Calculate basic features for correlation
eda_df['length_diff'] = abs(eda_df['q1_length'] - eda_df['q2_length'])
eda_df['word_diff'] = abs(eda_df['q1_words'] - eda_df['q2_words'])

# Correlation with target
correlation_features = ['q1_length', 'q2_length', 'q1_words', 'q2_words', 
                        'length_diff', 'word_diff', 'is_duplicate']
corr_df = eda_df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\nCorrelation with target variable (is_duplicate):")
target_corr = corr_df['is_duplicate'].sort_values(ascending=False)
for feature, corr_value in target_corr.items():
    if feature != 'is_duplicate':
        print(f"  {feature}: {corr_value:.4f}")

### 2.5.7 Summary Statistics
print("\n7. Summary Statistics:")
print("-" * 60)

summary_stats = eda_df.groupby('is_duplicate')[['q1_length', 'q2_length', 'q1_words', 'q2_words']].agg(['mean', 'median', 'std'])
print("\nStatistics by Duplicate Status:")
print(summary_stats)

print("\n" + "="*80)
print("✓ Exploratory Data Analysis Completed!")
print("="*80)

"""## 3. Text Preprocessing"""

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Comprehensive text preprocessing
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

print("Preprocessing questions...")
df['question1_processed'] = df['question1'].apply(preprocess_text)
df['question2_processed'] = df['question2'].apply(preprocess_text)

# Remove rows with empty processed questions
df = df[(df['question1_processed'] != '') & (df['question2_processed'] != '')]

print(f"✓ Preprocessing completed!")
print(f"Final dataset shape: {df.shape}")

# Show examples
print("\nExample:")
print(f"Original: {df.iloc[0]['question1']}")
print(f"Processed: {df.iloc[0]['question1_processed']}")

"""## 4. Advanced Feature Engineering"""

def calculate_basic_features(row):
    """Calculate basic text features"""
    q1 = row['question1']
    q2 = row['question2']
    q1_processed = row['question1_processed']
    q2_processed = row['question2_processed']

    # Length features
    q1_len = len(q1)
    q2_len = len(q2)
    q1_words = len(q1_processed.split())
    q2_words = len(q2_processed.split())

    # Difference features
    word_diff = abs(q1_words - q2_words)
    char_diff = abs(q1_len - q2_len)

    # Common words
    q1_tokens = set(q1_processed.split())
    q2_tokens = set(q2_processed.split())
    common_words = len(q1_tokens.intersection(q2_tokens))

    return pd.Series({
        'q1_len': q1_len,
        'q2_len': q2_len,
        'q1_words': q1_words,
        'q2_words': q2_words,
        'word_diff': word_diff,
        'char_diff': char_diff,
        'common_words': common_words
    })

def calculate_advanced_features(row):
    """Calculate advanced similarity features"""
    q1 = row['question1_processed']
    q2 = row['question2_processed']
    q1_orig = row['question1']
    q2_orig = row['question2']

    # Tokenize
    q1_tokens = set(q1.split())
    q2_tokens = set(q2.split())

    # Jaccard similarity
    if len(q1_tokens) == 0 and len(q2_tokens) == 0:
        jaccard = 1.0
    elif len(q1_tokens) == 0 or len(q2_tokens) == 0:
        jaccard = 0.0
    else:
        jaccard = len(q1_tokens.intersection(q2_tokens)) / len(q1_tokens.union(q2_tokens))

    # Word share ratio
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        word_share = 0.0
    else:
        word_share = len(q1_tokens.intersection(q2_tokens)) / (len(q1_tokens) + len(q2_tokens))

    # Sequence similarity
    seq_similarity = SequenceMatcher(None, q1, q2).ratio()

    # Fuzzy matching scores
    fuzz_ratio = fuzz.ratio(q1_orig, q2_orig) / 100.0
    fuzz_partial_ratio = fuzz.partial_ratio(q1_orig, q2_orig) / 100.0
    fuzz_token_sort = fuzz.token_sort_ratio(q1_orig, q2_orig) / 100.0
    fuzz_token_set = fuzz.token_set_ratio(q1_orig, q2_orig) / 100.0

    # Levenshtein distance
    levenshtein_dist = Levenshtein.distance(q1, q2)
    max_len = max(len(q1), len(q2))
    normalized_levenshtein = 1 - (levenshtein_dist / max_len) if max_len > 0 else 0

    # Cosine similarity
    try:
        vectorizer = CountVectorizer().fit([q1, q2])
        vectors = vectorizer.transform([q1, q2]).toarray()
        if len(vectors) == 2:
            norm_product = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
            cosine_sim = np.dot(vectors[0], vectors[1]) / (norm_product + 1e-10)
        else:
            cosine_sim = 0
    except:
        cosine_sim = 0

    return pd.Series({
        'jaccard_similarity': jaccard,
        'word_share': word_share,
        'sequence_similarity': seq_similarity,
        'fuzz_ratio': fuzz_ratio,
        'fuzz_partial_ratio': fuzz_partial_ratio,
        'fuzz_token_sort': fuzz_token_sort,
        'fuzz_token_set': fuzz_token_set,
        'levenshtein_normalized': normalized_levenshtein,
        'cosine_similarity': cosine_sim
    })

print("Calculating basic features...")
basic_features = df.apply(calculate_basic_features, axis=1)
df = pd.concat([df, basic_features], axis=1)

print("Calculating advanced features...")
advanced_features = df.apply(calculate_advanced_features, axis=1)
df = pd.concat([df, advanced_features], axis=1)

print("✓ Feature engineering completed!")
print(f"\nTotal features: {len(basic_features.columns) + len(advanced_features.columns)}")
print("\nFeature statistics:")
print(df[advanced_features.columns].describe())

"""## 5. Data Augmentation"""

def augment_data(df, augment_ratio=0.2):
    """
    Augment training data by swapping duplicate question pairs
    """
    print(f"Original dataset size: {len(df)}")

    # Swap questions for duplicate pairs
    duplicate_df = df[df['is_duplicate'] == 1].sample(frac=augment_ratio, random_state=42)

    augmented_rows = []
    for _, row in duplicate_df.iterrows():
        # Swap the questions
        new_row = row.copy()
        new_row['question1'] = row['question2']
        new_row['question2'] = row['question1']
        new_row['question1_processed'] = row['question2_processed']
        new_row['question2_processed'] = row['question1_processed']

        # Swap length/word features
        new_row['q1_len'] = row['q2_len']
        new_row['q2_len'] = row['q1_len']
        new_row['q1_words'] = row['q2_words']
        new_row['q2_words'] = row['q1_words']

        augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, augmented_df], ignore_index=True)

    print(f"Augmented dataset size: {len(result_df)}")
    print(f"Added {len(augmented_df)} augmented samples")

    return result_df

# Augment data (reduced for faster training)
df = augment_data(df, augment_ratio=0.05)  # Reduced from 0.15 to 0.05

# Use only 100% of data for faster training
print(f"Dataset before sampling: {len(df):,}")
df = df.sample(frac=1, random_state=42)
print(f"Dataset after 50% sampling: {len(df):,}")

"""## 6. Prepare Data for Deep Learning"""

# Parameters (optimized for speed)
MAX_VOCAB_SIZE = 30000  # Reduced from 50000
MAX_SEQUENCE_LENGTH = 15  # Reduced from 50 for faster processing

# Prepare features
X = df[['question1_processed', 'question2_processed']]
y = df['is_duplicate']

# Numerical features
feature_columns = ['q1_len', 'q2_len', 'q1_words', 'q2_words',
                   'word_diff', 'char_diff', 'common_words',
                   'jaccard_similarity', 'word_share', 'sequence_similarity',
                   'fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_token_sort',
                   'fuzz_token_set', 'levenshtein_normalized', 'cosine_similarity']

numerical_features = df[feature_columns].values

# Normalize features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

print(f"Number of numerical features: {numerical_features_scaled.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_train, num_test = train_test_split(
    numerical_features_scaled, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# Create tokenizer
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
all_questions = pd.concat([X_train['question1_processed'], X_train['question2_processed']])
tokenizer.fit_on_texts(all_questions)

vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
print(f"Vocabulary size: {vocab_size:,}")

# Convert to sequences
X_train_q1_seq = tokenizer.texts_to_sequences(X_train['question1_processed'])
X_train_q2_seq = tokenizer.texts_to_sequences(X_train['question2_processed'])
X_test_q1_seq = tokenizer.texts_to_sequences(X_test['question1_processed'])
X_test_q2_seq = tokenizer.texts_to_sequences(X_test['question2_processed'])

# Pad sequences
X_train_q1_pad = pad_sequences(X_train_q1_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_train_q2_pad = pad_sequences(X_train_q2_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_q1_pad = pad_sequences(X_test_q1_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_q2_pad = pad_sequences(X_test_q2_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print(f"\nPadded sequence shape: {X_train_q1_pad.shape}")
print("✓ Data preparation completed!")

"""## 7. Advanced Model Architecture"""

def create_advanced_model(vocab_size, num_features, embedding_dim=200, lstm_units=256):
    """
    Create advanced model with multiple merge strategies
    """
    # Input layers
    input_q1 = Input(shape=(MAX_SEQUENCE_LENGTH,), name='question1')
    input_q2 = Input(shape=(MAX_SEQUENCE_LENGTH,), name='question2')
    input_features = Input(shape=(num_features,), name='features')

    # Shared embedding layer
    embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name='shared_embedding'
    )

    # Shared LSTM layers
    lstm_layer1 = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='shared_lstm1'
    )
    lstm_layer2 = Bidirectional(
        LSTM(lstm_units//2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        name='shared_lstm2'
    )

    # Process Question 1
    embedded_q1 = embedding_layer(input_q1)
    lstm_q1 = lstm_layer1(embedded_q1)
    encoded_q1 = lstm_layer2(lstm_q1)

    # Process Question 2
    embedded_q2 = embedding_layer(input_q2)
    lstm_q2 = lstm_layer1(embedded_q2)
    encoded_q2 = lstm_layer2(lstm_q2)

    # Multiple merge strategies
    # 1. Concatenation
    concatenated = concatenate([encoded_q1, encoded_q2], name='concatenate')

    # 2. Absolute difference
    difference = Lambda(lambda x: K.abs(x[0] - x[1]), name='difference')([encoded_q1, encoded_q2])

    # 3. Element-wise multiplication
    multiplication = Multiply(name='multiply')([encoded_q1, encoded_q2])

    # 4. Cosine similarity
    dot_product = Lambda(
        lambda x: K.sum(x[0] * x[1], axis=1, keepdims=True),
        name='dot_product'
    )([encoded_q1, encoded_q2])

    # Merge all representations with numerical features
    merged = concatenate([
        concatenated,
        difference,
        multiplication,
        dot_product,
        input_features
    ], name='merge_all')

    # Dense layers with batch normalization
    dense1 = Dense(512, activation='relu', name='dense1')(merged)
    batch_norm1 = BatchNormalization(name='batch_norm1')(dense1)
    dropout1 = Dropout(0.4, name='dropout1')(batch_norm1)

    dense2 = Dense(256, activation='relu', name='dense2')(dropout1)
    batch_norm2 = BatchNormalization(name='batch_norm2')(dense2)
    dropout2 = Dropout(0.3, name='dropout2')(batch_norm2)

    dense3 = Dense(128, activation='relu', name='dense3')(dropout2)
    batch_norm3 = BatchNormalization(name='batch_norm3')(dense3)
    dropout3 = Dropout(0.2, name='dropout3')(batch_norm3)

    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(dropout3)

    # Create model
    model = Model(
        inputs=[input_q1, input_q2, input_features],
        outputs=output,
        name='advanced_siamese_model'
    )

    return model

# Create model (optimized parameters for speed)
print("Creating advanced model...")
advanced_model = create_advanced_model(
    vocab_size=vocab_size,
    num_features=num_train.shape[1],
    embedding_dim=128,  # Reduced from 200
    lstm_units=128  # Reduced from 256
)

# Compile model
advanced_model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("\nModel Architecture:")
advanced_model.summary()

print(f"\nTotal trainable parameters: {advanced_model.count_params():,}")

"""## 8. Train Advanced Model"""

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"Class weights: {class_weight_dict}")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=7,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.3,
    patience=3,
    min_lr=0.00001,
    mode='max',
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train model
print("\nTraining advanced model...")
print("="*70)

history = advanced_model.fit(
    [X_train_q1_pad, X_train_q2_pad, num_train],
    y_train,
    validation_split=0.15,
    epochs=5,  # Reduced for faster training
    batch_size=128,  # Increased from 64 for faster training
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

print("\n✓ Training completed!")

"""## 9. Visualize Training History"""

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# AUC
axes[1, 1].plot(history.history['auc'], label='Training AUC', linewidth=2)
axes[1, 1].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
axes[1, 1].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""## 10. Model Evaluation"""

# Make predictions
print("Making predictions on test set...")
y_pred_proba = advanced_model.predict(
    [X_test_q1_pad, X_test_q2_pad, num_test],
    verbose=0
).flatten()

y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*70)
print("ADVANCED MODEL EVALUATION")
print("="*70)
print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Duplicate', 'Duplicate']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Duplicate', 'Duplicate'],
            yticklabels=['Not Duplicate', 'Duplicate'])
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""## 11. Sample Predictions"""

def predict_similarity(question1, question2, model, tokenizer, scaler, threshold=0.5):
    """
    Predict if two questions are duplicates
    """
    # Preprocess
    q1_processed = preprocess_text(question1)
    q2_processed = preprocess_text(question2)

    # Convert to sequences
    q1_seq = tokenizer.texts_to_sequences([q1_processed])
    q2_seq = tokenizer.texts_to_sequences([q2_processed])

    # Pad
    q1_pad = pad_sequences(q1_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    q2_pad = pad_sequences(q2_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Calculate features
    q1_tokens = set(q1_processed.split())
    q2_tokens = set(q2_processed.split())

    jaccard = len(q1_tokens.intersection(q2_tokens)) / len(q1_tokens.union(q2_tokens)) if len(q1_tokens.union(q2_tokens)) > 0 else 0
    word_share = len(q1_tokens.intersection(q2_tokens)) / (len(q1_tokens) + len(q2_tokens)) if (len(q1_tokens) + len(q2_tokens)) > 0 else 0
    seq_sim = SequenceMatcher(None, q1_processed, q2_processed).ratio()

    features = np.array([[
        len(question1), len(question2),
        len(q1_processed.split()), len(q2_processed.split()),
        abs(len(q1_processed.split()) - len(q2_processed.split())),
        abs(len(question1) - len(question2)),
        len(q1_tokens.intersection(q2_tokens)),
        jaccard, word_share, seq_sim,
        fuzz.ratio(question1, question2) / 100.0,
        fuzz.partial_ratio(question1, question2) / 100.0,
        fuzz.token_sort_ratio(question1, question2) / 100.0,
        fuzz.token_set_ratio(question1, question2) / 100.0,
        1 - (Levenshtein.distance(q1_processed, q2_processed) / max(len(q1_processed), len(q2_processed))) if max(len(q1_processed), len(q2_processed)) > 0 else 0,
        0  # cosine similarity placeholder
    ]])

    features_scaled = scaler.transform(features)

    # Predict
    probability = model.predict([q1_pad, q2_pad, features_scaled], verbose=0)[0][0]
    prediction = 1 if probability > threshold else 0

    return prediction, probability

# Test cases
print("="*70)
print("TESTING PREDICTION FUNCTION")
print("="*70)

test_cases = [
    ("How do I learn Python?", "What's the best way to learn Python programming?"),
    ("What is machine learning?", "How do I bake a cake?"),
    ("How can I lose weight?", "What are effective weight loss strategies?"),
    ("What is the capital of France?", "What is the capital of Germany?"),
    ("How to reverse a string in Python?", "How to reverse a string in Python programming?")
]

for q1, q2 in test_cases:
    pred, prob = predict_similarity(q1, q2, advanced_model, tokenizer, scaler)
    print(f"\nQ1: {q1}")
    print(f"Q2: {q2}")
    print(f"Prediction: {'✓ Duplicate' if pred == 1 else '✗ Not Duplicate'}")
    print(f"Confidence: {prob:.2%}")
    print("-"*70)

"""## 12. Save Model and Artifacts"""

import pickle

# Save model
advanced_model.save('quora_advanced_model.keras')
print("✓ Model saved as 'quora_advanced_model.keras'")

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved as 'tokenizer.pkl'")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as 'scaler.pkl'")

# Save configuration
config = {
    'MAX_VOCAB_SIZE': MAX_VOCAB_SIZE,
    'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
    'vocab_size': vocab_size,
    'feature_columns': feature_columns
}

with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Configuration saved as 'config.pkl'")

print("\n✓ All artifacts saved successfully!")