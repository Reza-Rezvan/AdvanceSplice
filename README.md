# AdvanceSplice

AdvanceSplice is a method for accurate splice site prediction, integrating N-gram one-hot encoding and ensemble modeling. This repository contains the code and data used in the paper [AdvanceSplice: Integrating N-gram one-hot encoding and ensemble modeling for enhanced accuracy](https://www.sciencedirect.com/science/article/abs/pii/S1746809424000752).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Introduction

Accurate splice site prediction is critical in genomics, essential for understanding gene expression and disease-associated mutations. AdvanceSplice integrates two feature extraction approaches: N-gram One-hot Encoding and character-to-numerical encoding, and employs majority voting in Ensemble Modeling to enhance the accuracy of splice site prediction.

## Features

- N-gram One-hot Encoding for capturing essential patterns within DNA sequences.
- Character-to-numerical encoding to enrich feature analysis.
- Ensemble modeling using deep learning models specialized in processing image-like binary representations and sequence information.
- Improved accuracy in splice site prediction compared to existing models.

## Installation

To use AdvanceSplice, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AdvanceSplice.git
    ```
2. Navigate to the directory:
    ```bash
    cd AdvanceSplice
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset in the required format (refer to the Datasets section).
2. Run the preprocessing script to generate N-gram one-hot encoded and character-to-numerical encoded features.
    ```bash
    python preprocess.py
    ```
3. Train the ensemble models using the training script.
    ```bash
    python train.py
    ```
4. Evaluate the models using the evaluation script.
    ```bash
    python evaluate.py
    ```

## Datasets

The datasets used in this study include HS3D, Homo Sapiens, and A. Thaliana. Please refer to the data preparation section in the paper for detailed instructions on dataset preprocessing.

## Results

Comparisons with existing models on datasets such as HS3D, Homo Sapiens, and A. Thaliana indicate that AdvanceSplice identifies splice sites more effectively. Detailed results and performance metrics are provided in the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- Name: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]

