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
    'name': [
        'Theta', 'Zeta', 'Epsilon', 'Eta', 'Iota', 'Delta', 'Kappa', 'Gamma', 'Alpha', 'Beta',
        'Lambda', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega', 'Mu', 'Nu',
        'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi'
    ],
    'description': [
        'Theta is top-rated for safety.',
        'Zeta is an innovative solution.',
        'Epsilon is versatile and easy to use.',
        'Eta provides exceptional value.',
        'Iota is compact and convenient.',
        'Delta is designed for efficiency.',
        'Kappa stands out with a modern design.',
        'Gamma offers premium quality.',
        'Alpha is a durable and stylish product.',
        'Beta is reliable and affordable.',
        'Lambda combines elegance and functionality.',
        'Sigma is robust and built to last.',
        'Tau is perfect for everyday use.',
        'Upsilon provides unmatched versatility.',
        'Phi is known for its innovative design.',
        'Chi offers exceptional comfort.',
        'Psi is an eco-friendly product.',
        'Omega stands out with its sleek design.',
        'Mu is highly efficient and user-friendly.',
        'Nu delivers outstanding performance.',
        'Xi is a premium quality product.',
        'Omicron is compact and highly durable.',
        'Pi combines style and functionality.',
        'Rho is perfect for modern homes.',
        'Sigma offers great value for money.',
        'Tau is easy to use and highly reliable.',
        'Upsilon is a top choice for professionals.',
        'Phi features a contemporary design.',
        'Chi is safe and comfortable for babies.',
        'Psi is sustainable and eco-friendly.'
    ],
    'price': [
        34.15, 76.86, 65.94, 38.86, 24.36, 10.55, 16.84, 10.22, 39.58, 83.92,
        45.50, 55.60, 27.45, 60.25, 70.30, 33.80, 48.90, 52.75, 22.10, 47.85,
        65.40, 29.90, 40.75, 53.65, 55.60, 27.45, 60.25, 70.30, 33.80, 48.90
    ],
    'category': [
        'Baby Products', 'Fashion', 'Electronics', 'Home Goods', 'Electronics', 'Automotive', 'Fashion',
        'Electronics', 'Electronics', 'Home Goods', 'Fashion', 'Home Goods', 'Electronics', 'Automotive',
        'Fashion', 'Baby Products', 'Home Goods', 'Electronics', 'Automotive', 'Electronics',
        'Fashion', 'Electronics', 'Home Goods', 'Home Goods', 'Home Goods', 'Electronics', 'Automotive',
        'Fashion', 'Baby Products', 'Home Goods'
    ]
}


# Convert data to DataFrame
df = pd.DataFrame(data)

# Augment data
augmented_df = augment_data(
    df, augmentation_factor=0
)  # Augment each example by generating 5 new examples with synonym replacement

# Concatenate original and augmented data
augmented_data = pd.concat([df, augmented_df], ignore_index=True)

# Shuffle the dataframe
augmented_data = augmented_data.sample(frac=1).reset_index(drop=True)

# Save to CSV
augmented_data.to_csv("generated_data1.csv", index=False)

# Display augmented data
print(augmented_data)
