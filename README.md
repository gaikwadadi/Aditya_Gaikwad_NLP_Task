
# Educational Text NLP using RNNs (Classification & Next Word Generation)

## Overview

This project demonstrates the use of Recurrent Neural Networks (RNNs) with Keras for two Natural Language Processing (NLP) tasks:

1. **Educational Text Classification**  
   Classify educational text snippets into predefined categories: Math, Science, History, and English.

2. **Next Word Generation (Auto-completion)**  
   Generate the next 20 words based on a seed sentence using a model trained on an educational text corpus.

---

## 1. Text Classification

### Task

Given a short educational text snippet, the model classifies it into one of the four categories.

### Dataset

- A manually curated CSV file: `educational_text_classification.csv`
- Contains labeled short educational sentences under categories: Math, Science, History, and English.

### Preprocessing

- Label encoding using `LabelEncoder`
- Tokenization and padding using `Tokenizer` and `pad_sequences`
- Pre-trained word embeddings using GloVe (100-dimensional vectors)
- Stratified 80/20 train-test split
- Computation of class weights to handle class imbalance

### Model Architecture

- `Embedding` layer using GloVe vectors
- Two-layer `Bidirectional GRU`
- `LayerNormalization` and `Dropout` for regularization
- `Dense` layer with softmax for multi-class classification

### Training Strategy

- Loss: `Sparse Categorical Crossentropy`
- Optimizer: `Adam`
- Metrics: `Accuracy`
- Callbacks: EarlyStopping, ReduceLROnPlateau

### Evaluation

- Accuracy and loss on the test set
- Confusion matrix and classification report
- Weighted F1 score

---

## 2. Next Word Generation

### Task

Given a sentence (at least 10 words), the model generates the next 20 words based on learned patterns from educational content.

### Dataset

- A long educational corpus (`.txt` file): `large_next_word_generation_corpus.txt`
- Text is lowercased and split into sentences using regex

### Preprocessing

- Tokenization and sequence creation from sentences
- Input: sequences of words
- Target: the next word in each sequence
- Padding to ensure uniform input shape

### Model Architecture

- `Embedding` layer
- Two stacked `LSTM` layers with dropout
- Final `Dense` layer with softmax activation

### Training Strategy

- Loss: `Sparse Categorical Crossentropy`
- Optimizer: `Adam`
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Text Generation

- Top-k sampling with temperature control
- Generates 20 words from a given seed text

---

## How to Run

### Prerequisites

- Python 3.7+
- TensorFlow
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/edu-nlp-rnn.git
   cd edu-nlp-rnn
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download GloVe embeddings (100d) if not already present:
   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

4. Run classification model:
   ```bash
   python classification_model.py
   ```

5. Run next word generation model:
   ```bash
   python next_word_generation.py
   ```

---

## Results

### Classification

- Test Accuracy: Reported after model evaluation
- Confusion Matrix and F1 Score included in output

### Next Word Generation

- Accepts a seed sentence
- Generates coherent next 20 words
- Example:

  ```
  Input: "Mathematics is the study of"
  Output: "symbols and the rules for manipulating those symbols the systematic science is in the form of explanations and predictions"
  ```

---

## Project Structure

```
.
├── classification_model.py
├── next_word_generation.py
├── educational_text_classification.csv
├── large_next_word_generation_corpus.txt
├── glove.6B.100d.txt
├── tokenizer.pkl
├── label_encoder.pkl
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Experiment with transformer-based models (e.g., BERT for classification)
- Use beam search or nucleus sampling for more fluent text generation
- Add web interface or notebook demo
