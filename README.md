# Melanoma Classification

This project focuses on building a deep learning pipeline to classify skin lesion images as malignant or benign using ResNet50. The final model achieves a respectable F1 score on a balanced dataset.

## Overview

- Binary classification task for melanoma detection
- Utilizes a modified dataset with a 50-50 class split
- Early experiments implemented in `main.py`
- Final version of the project, with refined results and experiments, is available in `AML_main.ipynb`

## Features

- ResNet50 model adapted for binary classification
- Custom `ImageDataset` class supporting JPEG and DICOM images
- Data augmentation and preprocessing with `torchvision.transforms`
- Balanced training set for improved generalization
- Tracks training and testing metrics including F1 score

## Project Structure

```
project/
│
├── main.py              # Initial version with full training pipeline
├── AML_main.ipynb       # Final and refined version of the project
├── modified.csv         # Balanced dataset CSV [not included here]
├── train/               # Folder containing JPEG and/or DICOM images
└── models/              # Folder for saved models (optional)
```

## Setup

1. Clone the repository and install dependencies:

```bash
pip install torch torchvision pandas scikit-learn pydicom tqdm pillow
```

2. Prepare the dataset:
   - Use a CSV file with image IDs and labels
   - Place all JPEG images inside the same folder as DICOMs if using both types
   - Ensure paths are updated in the scripts

3. Run the notebook or script:
   - Use `AML_main.ipynb` for exploration and final evaluation
   - Use `main.py` for a scripted pipeline

## Results

- **Model**: ResNet50 (pretrained)
- **Epochs**: 10
- **Final Test Accuracy**: 73.5%
- **Final F1 Score**: 0.75

## Future Work

- Try other architectures like EfficientNet or DenseNet
- Tune hyperparameters such as learning rate and batch size
- Add more data augmentation
- Improve DICOM preprocessing

## Notes

- Tested on macOS with MPS backend, CUDA is also supported
- Dataset and trained models are not included due to size constraints

## License

This project is for educational purposes only.
