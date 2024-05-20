import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from random import choice


# Function to find synonyms of a word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


# Function to perform synonym replacement
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_sentence = sentence
    replacements = set()  # To track replacements and avoid repetitions
    for _ in range(n):
        word = choice(words)
        if word not in replacements:
            replacements.add(word)
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = choice(synonyms)
                new_sentence = new_sentence.replace(word, synonym, 1)
    return new_sentence


# Generate augmented data
def augment_data(data, augmentation_factor=1):
    augmented_data = []
    for _, row in data.iterrows():
        original_description = row["description"]
        augmented_descriptions = set()  # To avoid duplicates
        for _ in range(augmentation_factor):
            augmented_description = synonym_replacement(original_description)
            augmented_descriptions.add(augmented_description)
        for desc in augmented_descriptions:
            augmented_data.append(
                {
                    "name": row["name"],
                    "description": desc,
                    "price": row["price"],
                    "category": row["category"],
                }
            )
    return pd.DataFrame(augmented_data)


# Original data
data = {
    "name": [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
    ],
    "description": [
        "Alpha is a durable and stylish product.",
        "Beta is reliable and affordable.",
        "Gamma offers premium quality.",
        "Delta is designed for efficiency.",
        "Epsilon is versatile and easy to use.",
        "Zeta is an innovative solution.",
        "Eta provides exceptional value.",
        "Theta is top-rated for safety.",
        "Iota is compact and convenient.",
        "Kappa stands out with a modern design.",
    ],
    "price": np.random.uniform(10, 100, size=10).round(2),
    "category": [
        "Electronics",
        "Home Goods",
        "Electronics",
        "Automotive",
        "Electronics",
        "Fashion",
        "Home Goods",
        "Baby Products",
        "Electronics",
        "Fashion",
    ],
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Augment data
augmented_df = augment_data(
    df, augmentation_factor=5
)  # Augment each example by generating 5 new examples with synonym replacement

# Concatenate original and augmented data
augmented_data = pd.concat([df, augmented_df], ignore_index=True)

# Shuffle the dataframe
augmented_data = augmented_data.sample(frac=1).reset_index(drop=True)

# Save to CSV
augmented_data.to_csv("generated_data1.csv", index=False)

# Display augmented data
print(augmented_data)
