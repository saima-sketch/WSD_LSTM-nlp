"""
model.py
--------
Trains an LSTM model for Word Sense Disambiguation of the word "bank".
Class 0 = Financial Bank
Class 1 = River Bank
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ------------------------------------------------------------------
# 1. Training data (exactly as given in the experiment)
# ------------------------------------------------------------------
sentences = [
    "I deposited money in the bank",
    "The bank gave me a loan",
    "I went to the bank to withdraw cash",
    "The river bank was beautiful",
    "We sat on the bank of the river",
    "The children played near the river bank"
]

labels = [0, 0, 0, 1, 1, 1]  # 0 = Financial Bank, 1 = River Bank

# ------------------------------------------------------------------
# 2. Tokenizer + sequences + padding
# ------------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

# Use the longest sequence length from training for padding
MAX_LEN = max(len(seq) for seq in sequences)

padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# ------------------------------------------------------------------
# 3. Build the LSTM model
# ------------------------------------------------------------------
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16),
    LSTM(32),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------------
# 4. Train the model (once, at import time)
# ------------------------------------------------------------------
print("[model.py] Training LSTM model for WSD...")
model.fit(
    padded,
    np.array(labels),
    epochs=100,
    verbose=0
)
print("[model.py] Model training complete.")

# ------------------------------------------------------------------
# 5. Prediction function
# ------------------------------------------------------------------
def predict_sense(sentence: str) -> dict:
    """
    Predict the sense of the word 'bank' in the given sentence.

    Returns a dict:
    {
        "class": 0 or 1,
        "meaning": "Financial Bank" or "River Bank",
        "probabilities": {
            "financial_bank": float,
            "river_bank": float
        }
    }
    """
    # Convert sentence to sequence using the SAME tokenizer
    seq = tokenizer.texts_to_sequences([sentence])

    # Apply the SAME padding as training
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # Get prediction probabilities
    probs = model.predict(padded_seq, verbose=0)[0]  # shape (2,)

    # argmax to get class
    class_id = int(np.argmax(probs))

    meaning = "Financial Bank" if class_id == 0 else "River Bank"

    return {
        "class": class_id,
        "meaning": meaning,
        "probabilities": {
            "financial_bank": float(probs[0]),
            "river_bank": float(probs[1])
        }
    }