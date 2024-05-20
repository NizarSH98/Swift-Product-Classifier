import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)  # Update the filepath if necessary
    return df["description"], df["category"]


def load_tools():
    # Load tokenizer and label classes
    with open("tokenizer.json", "r") as file:
        tokenizer = tokenizer_from_json(file.read())
    label_classes = np.load("label_classes.npy", allow_pickle=True)
    return tokenizer, label_classes


def load_my_model(filepath):
    # Load the trained model
    model = tf_load_model(filepath)  # Update the filepath if necessary
    return model


def predict_categories(descriptions, tokenizer, model, max_length):
    # Tokenize and pad descriptions
    seq = tokenizer.texts_to_sequences(descriptions)
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    # Predict categories
    predictions = model.predict(padded)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes


def evaluate_model(y_true, y_pred, label_classes):
    # Convert numerical labels in y_pred to string labels
    y_pred_labels = [label_classes[label] for label in y_pred]

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    report = classification_report(y_true, y_pred_labels, target_names=label_classes)
    return accuracy, report


if __name__ == "__main__":
    # Load data
    X_test, y_test = load_data(
        "generated_data1.csv"
    )  # Update the filepath if necessary

    # Load tools
    tokenizer, label_classes = load_tools()

    # Load model
    model = load_my_model(
        "product_classifier_model.keras"
    )  # Update the filepath if necessary
    max_length = int(open("max_length.txt", "r").read())  # Load max_length

    # Predict categories
    y_pred = predict_categories(X_test, tokenizer, model, max_length)

    # Evaluate model
    accuracy, report = evaluate_model(y_test, y_pred, label_classes)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(report)
