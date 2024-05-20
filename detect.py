import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_tools():

    with open("tokenizer.json", "r") as file:
        tokenizer = tokenizer_from_json(file.read())
    label_classes = np.load("label_classes.npy", allow_pickle=True)
    model = load_model("product_classifier_model.keras")
    with open("max_length.txt", "r") as f:
        max_length = int(f.read())
    return tokenizer, label_classes, model, max_length


def predict_category(description, tokenizer, label_classes, model, max_length):
    seq = tokenizer.texts_to_sequences([description])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    category_idx = np.argmax(prediction)
    return label_classes[category_idx]


def main():
    test_descriptions = [
        "This is a durable and stylish electronic device.",
        "The new fashion accessory is trendy and affordable.",
        "I am looking for a reliable and efficient home appliance.",
        "This innovative gadget offers premium features.",
        "The automotive tool is designed for safety and performance.",
        "I need a versatile and easy-to-use kitchen appliance.",
        "This toy provides exceptional value and entertainment.",
        "The baby product is top-rated for safety and comfort.",
        "This compact device is perfect for everyday use.",
        "The fashion item stands out with its modern design.",
    ]

    # Load tools
    tokenizer, label_classes, model, max_length = load_tools()

    # Test the model with test descriptions
    print("Testing with pre-defined descriptions:")
    for description in test_descriptions:
        predicted_category = predict_category(
            description, tokenizer, label_classes, model, max_length
        )
        print(f"Description: {description}")
        print(f"Predicted Category: {predicted_category}\n")

    # Prompt the user to classify descriptions
    print("Now let's classify your descriptions:")
    while True:
        user_input = input("Enter a product description (or '0' to exit): ")
        # Check if the user wants to exit
        if user_input.strip() == "0":
            print("Exiting...")
            break
        # Predict category
        predicted_category = predict_category(
            user_input, tokenizer, label_classes, model, max_length
        )
        print(f"Predicted Category: {predicted_category}\n")


if __name__ == "__main__":
    main()
