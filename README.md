# Aditya_Gaikwad_NLP_Task
Objective: This assignment aims to evaluate your understanding and practical implementation skills related to Recurrent Neural Networks (RNNs) for Natural Language Processing (NLP) tasks using the Keras library (TensorFlow backend 1. 2. preferred).


# Educational Text Classification and Next Word Generation

This repository contains two primary NLP tasks:

1. **Educational Text Classification** - A Bidirectional GRU-based neural network for classifying educational text into categories: Math, Science, History, and English.
2. **Next Word Generation** - A Simple RNN-based model to generate contextually relevant text for scientific concepts.

## Table of Contents

* [Project Structure](#project-structure)
* [Datasets](#datasets)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architectures](#model-architectures)
* [Training Details](#training-details)
* [Evaluation](#evaluation)
* [Text Generation](#text-generation)
* [Results](#results)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)

## Project Structure

```
📂 project-root
│
├── educational_text_classification.csv         # Dataset for text classification
├── next_word_generation.csv                    # Dataset for text generation
│
├── model_classification.h5                    # Trained classification model
├── model_generation.h5                        # Trained generation model
│
├── tokenizer.pkl                              # Tokenizer for classification model
├── label_encoder.pkl                          # Label encoder for classification labels
│
├── main.py                                    # Main script for running models
├── utils.py                                   # Helper functions for data preprocessing
│
├── README.md                                  # Project documentation
└── requirements.txt                           # Dependencies
```

## Datasets

1. **Educational Text Classification Dataset**

   * Contains labeled educational text in four categories: Math, Science, History, and English.

2. **Next Word Generation Dataset**

   * Contains scientific sentences and phrases for building next-word predictions.

## Installation

To set up the environment, run:

```bash
pip install -r requirements.txt
```

## Usage

For Educational Text Classification:

```bash
python main.py --mode classification --input "Newton's laws of motion explain the relationship between a body and the forces acting upon it."
```

For Next Word Generation:

```bash
python main.py --mode generation --input "photosynthesis is the process"
```

## Model Architectures

1. **Educational Text Classification**

   * Bidirectional GRU layers with Layer Normalization and Dense layers for multi-class classification.

2. **Next Word Generation**

   * SimpleRNN architecture with an Embedding layer to generate contextual next words based on input text.

## Training Details

* **Classification Model**:

  * K-Fold Cross-Validation (5 splits)
  * Class Weight Adjustment for imbalance
  * Learning Rate Scheduling
  * Early Stopping and Model Checkpointing

* **Next Word Generation Model**:

  * Early Stopping based on loss
  * Temperature-based text generation

## Evaluation

* **Confusion Matrix** and **Classification Report** for classification.
* Accuracy metrics are displayed during training.

## Text Generation

Example:

```
Input: "Photosynthesis is the process"
Generated: "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water..."
```

## Results

* Achieved high accuracy on educational text classification.
* Generated coherent scientific text sequences for the generation task.

## Future Work

* Expand datasets to include more granular educational subjects.
* Improve generation coherence with larger RNN architectures.

## Contributing

Feel free to fork the repository and raise PRs for improvements!

## License

MIT License

