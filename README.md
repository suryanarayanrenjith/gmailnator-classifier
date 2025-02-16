# Gmailnator Classifier

## Overview
This repository contains code for training and testing a machine learning model designed specifically to classify Gmail addresses provided by [Gmailnator](https://www.emailnator.com). The goal is to detect Gmail addresses used for signups that attempt to bypass disposable email checks. **Note:** This classifier works exclusively with `gmail.com` and `googlemail.com` domains.

## Features
- **Specialized for Gmail/Googlemail:**  
  Focused solely on Gmail addresses to detect potentially disposable Gmailnator emails.
- **Advanced Feature Engineering:**  
  Extracts 15 detailed features including:
  - Domain flags (checks for `gmail.com` and `googlemail.com`)
  - Local-part analysis (e.g., presence of `+`, number of segments, digit-to-letter ratio, entropy)
  - Additional metrics (e.g., count of underscores, hyphens, and special characters)
- **Robust Neural Network Architecture:**  
  Utilizes multiple hidden layers with Batch Normalization, Dropout, Early Stopping, and Model Checkpointing for improved generalization and prevention of overfitting.

## Technical Specifications
- **Domain:** Gmail addresses (`gmail.com` and `googlemail.com`) only.
- **Python Version:** 3.10.10
- **TensorFlow:** 2.13.0 (integrated Keras)
- **scikit-learn:** 1.3.0
- **Joblib:** 1.4.2
- **Numpy:** 1.24.2

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/suryanarayanrenjith/gmailnator-classifier.git
cd gmailnator-classifier
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
```

- **On macOS/Linux:**
    
    ```bash
    source venv/bin/activate
    ```
    
- **On Windows:**
    
    ```bash
    venv\Scripts\activate
    ```
    

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

### Data Preparation

Prepare or use the training data file (e.g., `training_data.txt`) with each line formatted as:

```
email_address,label
```

- **Example:**
    
    ```
    example@gmail.com,1
    legitimate@gmail.com,0
    ```
    

Here, `1` indicates that the Gmail address is considered disposable (potentially provided by Gmailnator) and `0` indicates a legitimate Gmail address.

### Train the Model

Run the following command to train the model:

```bash
python train_model.py training_data.txt --epochs 100 --batch_size 32
```

After training, the best model will be saved as `model.h5` and the feature scaler as `scaler.save` inside the `models` folder.

## Testing the Model

The testing script (`test_model.py`) loads the saved `model.h5` and `scaler.save`, extracts the 15 engineered features from the Gmail address, and then classifies the input.

### To Test a Single Email

```bash
python test_model.py --email user@gmail.com
```

### To Test Multiple Emails

Create a text file (e.g., `test_emails.txt`) with one Gmail address per line, then run:

```bash
python test_model.py --file test_emails.txt
```

## Model Accuracy

Based on our internal experiments with a held-out validation set, the refined model achieved an estimated validation accuracy of **93%**.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.
