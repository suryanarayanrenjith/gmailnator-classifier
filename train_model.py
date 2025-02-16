#!/usr/bin/env python3
"""
Train a disposable email classifier using a text file input.
Each line in the training file should be in the format:
    email_address,label
where label is 1 for disposable and 0 for legitimate.
Tested on Python 3.10.10.
"""

import argparse
import re
import math
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def compute_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    length = len(s)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_features(email: str) -> np.ndarray:
    """
    Extract features from an email address.
    
    Engineered features (total 15):
      0.  is_gmail:           1 if domain is 'gmail.com', else 0.
      1.  is_googlemail:      1 if domain is 'googlemail.com', else 0.
      2.  plus_present:       1 if '+' is found in the local part.
      3.  num_segments:       Number of dot-separated segments in the local part.
      4.  avg_seg_length:     Average length of each segment.
      5.  digit_letter_ratio: Ratio of digits to letters in the local part.
      6.  local_length:       Total length of the local part.
      7.  count_digits:       Number of digit characters in local part.
      8.  count_letters:      Number of letter characters in local part.
      9.  num_dots:           Number of dots (num_segments - 1).
      10. special_count:      Count of special characters (excluding a-z, 0-9, dot, plus, underscore, hyphen).
      11. domain_length:      Length of the domain.
      12. count_underscores:  Number of underscores in the local part.
      13. count_hyphens:      Number of hyphens in the local part.
      14. local_entropy:      Shannon entropy of the local part.
    """
    email = email.strip().lower()
    try:
        local, domain = email.split('@')
    except ValueError:
        return np.zeros(15)
    
    is_gmail = 1 if domain == "gmail.com" else 0
    is_googlemail = 1 if domain == "googlemail.com" else 0
    plus_present = 1 if '+' in local else 0

    segments = local.split('.')
    num_segments = len(segments)
    avg_seg_length = np.mean([len(seg) for seg in segments]) if segments else 0

    count_digits = len(re.findall(r'\d', local))
    count_letters = len(re.findall(r'[a-z]', local))
    digit_letter_ratio = count_digits / count_letters if count_letters > 0 else 0
    local_length = len(local)
    num_dots = num_segments - 1
    special_count = len(re.findall(r'[^a-z0-9.+_-]', local))

    domain_length = len(domain)
    count_underscores = local.count('_')
    count_hyphens = local.count('-')
    local_entropy = compute_entropy(local)
    
    features = np.array([
        is_gmail,
        is_googlemail,
        plus_present,
        num_segments,
        avg_seg_length,
        digit_letter_ratio,
        local_length,
        count_digits,
        count_letters,
        num_dots,
        special_count,
        domain_length,
        count_underscores,
        count_hyphens,
        local_entropy
    ])
    return features


def create_feature_dataset(emails: list[str]) -> np.ndarray:
    return np.array([extract_features(email) for email in emails])


def main():
    parser = argparse.ArgumentParser(
        description="Train a disposable email classifier and save the model as model.h5"
    )
    parser.add_argument("text_file", 
                        help="Path to a text file with lines formatted as: email_address,label")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    args = parser.parse_args()

    emails = []
    labels = []

    with open(args.text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                email, label_str = line.split(",", 1)
                label = int(label_str.strip())
                emails.append(email.strip())
                labels.append(label)
            except Exception as e:
                print(f"Skipping line: {line}. Error: {e}")
                continue

    if not emails:
        print("No valid data found in the file.")
        return

    X = create_feature_dataset(emails)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(cls): weight for cls, weight in zip(classes, class_weights)}

    input_dim = X_train_scaled.shape[1]
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("models/best_model.h5", monitor="val_loss", save_best_only=True)

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    model.save("models/model.h5")
    joblib.dump(scaler, "models/scaler.save")
    print("Model saved as model.h5 and scaler saved as scaler.save.")


if __name__ == "__main__":
    main()