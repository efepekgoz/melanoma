
# Melanoma Classification

This project applies deep learning techniques to classify melanoma images using a fine-tuned ResNet50 model in PyTorch. The dataset consists of balanced malignant and benign skin lesions, with preprocessing and augmentation support for both DICOM and JPEG formats.

---

## 🔍 Overview

- **Model:** ResNet50 (pretrained on ImageNet)
- **Task:** Binary classification (malignant vs benign)
- **Input format:** JPEG and DICOM (.dcm)
- **Evaluation:** Accuracy, F1 Score

---


## 🧪 Requirements

Install Python packages:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchvision
pandas
numpy
pydicom
Pillow
scikit-learn
tqdm
```

---

## ⚙️ Training Pipeline

1. **Data Preparation**  
   - Reads from `modified.csv`
   - Matches JPEG or DICOM paths automatically
   - Applies resizing, normalization, and tensor conversion

2. **Model Definition**  
   - Uses pretrained ResNet50
   - Replaces final layer for binary output

3. **Training**  
   - Uses Adam optimizer and CrossEntropy loss
   - Trains for 10 epochs (default)

4. **Evaluation**  
   - Reports accuracy, loss, and F1 score on test set (20% split)

---

## 🧾 Usage

1. **Modify Paths**

Update paths in the script to reflect your file system:
```python
csv_file='/path/to/modified.csv'
root_dir='/path/to/train/'
```

2. **Run Script**

```bash
python main.py
```

---

## 📊 Results

| Metric        | Value     |
|---------------|-----------|
| Train Accuracy | ~72.3%    |
| Test Accuracy  | ~73.5%    |
| F1 Score       | ~0.75     |

*Tested on balanced 50/50 dataset with 10 epochs and batch size 4.*

---

## 📌 TODO

- Experiment with different models (e.g. EfficientNet, DenseNet)
- Add DICOM support back in and augment pipeline
- Improve transforms and data augmentation
- Hyperparameter tuning: LR, batch size, epochs

---

## 📜 License

This project is under the MIT License.
