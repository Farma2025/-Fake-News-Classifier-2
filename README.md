# Fake News Detection with RNN and LSTM 🕵️‍♂️📰

## Overview

This project implements a sophisticated Fake News Detection system using two deep learning models: Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM). The application provides an interactive Streamlit web interface to classify news articles as potentially fake or credible.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

Try the [Streamlit app](https://fake-news-detection-with-rnn-and-lstm.streamlit.app/)! 

![alt text](image.png)
## 🚀 Features

- **Two Advanced Machine Learning Models**
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)

- **Interactive Streamlit Web Interface**
  - Easy text input for news article classification
  - Model performance visualization
  - Detailed model insights

- **Comprehensive Text Preprocessing**
  - Lowercase conversion
  - Number removal
  - Punctuation elimination
  - Stopword removal
  - Lemmatization

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip

### Clone the Repository
```bash
git clone https://github.com/zyad-alsharnobi/Fake_News_Detection_with_RNN_and_LSTM.git
cd Fake_News_Detection_with_RNN_and_LSTM
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

### Training the Models
```bash
python development.ipynb
```

### Launching the Streamlit App
```bash
streamlit run app.py
```

## 📊 Model Performance

### RNN Model
- Test Accuracy: 85.65%

### LSTM Model
- Test Accuracy: 97.25%

## 🧠 Model Architecture

### RNN Model
- Embedding Layer (32 dimensions)
- Two SimpleRNN Layers (64 and 32 units)
- Dropout Layers
- Sigmoid Output Layer

### LSTM Model
- Embedding Layer (32 dimensions)
- Two LSTM Layers (64 and 32 units)
- Dropout Layers
- Sigmoid Output Layer

## 📝 Preprocessing Steps
1. Convert text to lowercase
2. Remove numbers
3. Remove punctuation
4. Tokenize text
5. Remove stopwords
6. Lemmatize words

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments
- TensorFlow
- NLTK
- Streamlit
- Scikit-learn

