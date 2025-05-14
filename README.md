# AdvanceSplice: Integrating N-gram one-hot encoding and ensemble modeling for enhanced accuracy

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
2. Use the Deep_models encoder first ro encode your data (AdvanceSplice used One-hot encoding). if you want to use two or three gram data, just read data because we encode and save them in Data.
3. Use LSTM_Char2Num encoder (Char2int) to use Character-to-numerical encoding.
4. Train your data with deep models (Deep_models or LSTM_Char2Num, based on your encoder method).
5. Evaluate the models.


## Datasets

The datasets used in this study include HS3D, Homo Sapiens, and A. Thaliana. Please refer to the data preparation section in the paper for detailed instructions on dataset preprocessing.

## Results

Comparisons with existing models on datasets such as HS3D, Homo Sapiens, and A. Thaliana indicate that AdvanceSplice identifies splice sites more effectively. Detailed results and performance metrics are provided in the paper.


## Contact

For any questions or inquiries, please contact:
reza.rzvn1@gmail.com

