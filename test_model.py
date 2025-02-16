#!/usr/bin/env python3
"""
Test the disposable email classifier using the saved model.h5 and scaler.
Works with Python 3.10.10.

Usage:
    To test a single email:
        python test_model.py --email user@example.com
    To test multiple emails from a text file:
        python test_model.py --file test_emails.txt
"""

import argparse
import re
import math
import numpy as np
import joblib
from tensorflow.keras.models import load_model

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
    Extract features from an email address using the refined feature engineering.

    Features (15 total):
      0.  is_gmail:           1 if domain is 'gmail.com', else 0.
      1.  is_googlemail:      1 if domain is 'googlemail.com', else 0.
      2.  plus_present:       1 if '+' is found in the local part.
      3.  num_segments:       Number of dot-separated segments in the local part.
      4.  avg_seg_length:     Average length of each segment.
      5.  digit_letter_ratio: Ratio of digits to letters in the local part.
      6.  local_length:       Total length of the local part.
      7.  count_digits:       Number of digit characters in the local part.
      8.  count_letters:      Number of letter characters in the local part.
      9.  num_dots:           Number of dots (num_segments - 1).
     10.  special_count:      Count of special characters (excluding a-z, 0-9, dot, plus, underscore, hyphen).
     11.  domain_length:      Length of the domain.
     12.  count_underscores:  Number of underscores in the local part.
     13.  count_hyphens:      Number of hyphens in the local part.
     14.  local_entropy:      Shannon entropy of the local part.
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

def main():
    parser = argparse.ArgumentParser(description="Test the improved disposable email classifier model.")
    parser.add_argument("--email", type=str, help="A single email address to test.")
    parser.add_argument("--file", type=str, help="Path to a text file with one email per line.")
    args = parser.parse_args()

    try:
        model = load_model("models/model.h5")
        scaler = joblib.load("models/scaler.save")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    emails = []
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                emails = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.email:
        emails = [args.email.strip()]
    else:
        user_input = input("Enter an email address to test: ").strip()
        emails = [user_input]

    for email in emails:
        features = extract_features(email)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0][0]
        label = "Disposable" if prediction >= 0.5 else "Legitimate"
        print(f"{email}: {label} (score: {prediction:.4f})")

if __name__ == "__main__":
    main()