"""
AI Engineer (m/f/x) Assessment at payever

Documentation
Approach Taken:

Data Generation: Generated a synthetic dataset of products
with attributes like name, description, price, and category using Python.

Data Preprocessing: Tokenized the descriptions and encoded the categories
to prepare the data for training a machine learning model.

Model Development: Built a simple neural network model using TensorFlow
to classify product categories based on text descriptions. The model consists
 of an embedding layer, a global average pooling layer, and dense layers.

Training and Evaluation: Trained the model with the generated dataset
and evaluated using accuracy as the metric.

Assumptions:

    The text descriptions sufficiently describe the product categories.

    Categories are balanced in distribution (not necessarily true in
    real-world scenarios).

Limitations:

    The dataset is very small and synthetic, which might not represent
    real-world complexities and variances.

    Model simplicity might not capture complex patterns in data, thus may
    underperform on more diverse and large datasets.
"""

# train.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Unused import removed
# import numpy as np
# import tensorflow as tf


def preprocess_data(df):
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(df["description"])
    tokenized = tokenizer.texts_to_sequences(df["description"])
    max_length = max(len(x) for x in tokenized)
    padded = pad_sequences(tokenized, maxlen=max_length, padding="post")

    label_encoder = LabelEncoder()
    categories_encoded = label_encoder.fit_transform(df["category"])

    return padded, categories_encoded, tokenizer, label_encoder, max_length


def build_model(input_length):
    model = Sequential(
        [
            Embedding(input_dim=100, output_dim=16, input_length=input_length),
            GlobalAveragePooling1D(),
            Dense(10, activation="relu"),
            Dense(len(df["category"].unique()), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Load data
df = pd.read_csv("generated_data1.csv")

padded_descriptions, category_encoded, tokenizer, label_encoder, max_len = (
    preprocess_data(df)
)

# Build and train the model
model = build_model(max_len)
model.fit(padded_descriptions, category_encoded, epochs=10, validation_split=0.2)

# Save the model and preprocessing tools
model.save("product_classifier_model.keras")

tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as file:
    file.write(tokenizer_json)
label_encoder.classes_.dump("label_classes.npy")
with open("max_length.txt", "w") as f:
    f.write(str(max_len))
