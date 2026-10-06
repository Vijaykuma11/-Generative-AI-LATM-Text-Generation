# Generative-AI-LSTM-Text-Generation

## Overview
LSTM based text generator trained on Shakespeare dataset. Generates coherent text from seed input.

## Dataset
- Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Public Domain - Project Gutenberg
- 1M characters, .txt format

## Model Architecture
- Embedding Layer (100 dim)
- LSTM 150 units + Dropout 0.2
- LSTM 150 units + Dropout 0.2
- Dense Softmax

Loss: categorical_crossentropy, Optimizer: Adam
Train/Val: 90/10, EarlyStopping patience=3

## How to Run
pip install -r requirements.txt
python main.py

## Generated Output Examples
Seed: "the king said to his" -> "the king said to his servants and the people shall be..."
Seed: "to be or not to" -> "to be or not to be the true love of the heart..."
Seed: "love is a beautiful" -> "love is a beautiful thing and the soul..."

## Bonus Experiments
- Seq 30, 1xLSTM 128: Fast but incoherent
- Seq 50, 2xLSTM 150 + Dropout: Best - Coherent
- Seq 100, 2xLSTM 200: Slow, slight overfit

## Author
Vijay Kumar Gupta
