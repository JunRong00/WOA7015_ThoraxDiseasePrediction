# Thorax Disease Prediction using CNN Models
This project focuses on building and evaluating convolutional neural network (CNN) models for the prediction of thoracic diseases using the NIH Chest X-rays Dataset. The task involves multi-label classification, where each chest X-ray can exhibit one or more thoracic diseases such as pneumonia, effusion, or cardiomegaly. The goal is to identify the best-performing model and techniques for accurate and efficient disease prediction.

## Dataset
The dataset used for this project is the NIH Chest X-rays Dataset, a publicly available dataset containing over 112,000 frontal-view X-ray images of 30,805 unique patients. Each image is labeled with up to 14 thoracic diseases. The dataset also includes patient demographics and additional metadata.

Dataset Source: [NIH Chest X-rays Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)

## CNN Model Architecture
Four CNN models were implemented and evaluated to identify the best architecture for thorax disease prediction:

1. MobileNet without Pre-trained Weights:
A lightweight CNN trained from scratch on the NIH dataset.
Relies solely on the dataset for learning feature representations.

2. MobileNet with ImageNet Pre-trained Weights:
Uses MobileNet initialized with ImageNet weights for transfer learning.
Fine-tuned for the thorax disease classification task, leveraging features learned from a large, general-purpose dataset.

3. Custom CNN Model:
A bespoke architecture designed specifically for this task.
Includes multiple convolutional, pooling, and dropout layers tailored for feature extraction and regularization.

4. DenseNet with CheXNet Pre-trained Weights:
DenseNet121 pre-trained on the CheXNet dataset, designed explicitly for medical imaging tasks.
Fine-tuned to adapt the medical-specific feature representations for the NIH dataset.

## Techniques Used
1. Data Preprocessing:
- Images resized to a uniform input shape (e.g., 128×128 pixels).
- Grayscale conversion to reduce complexity where applicable.
- Augmentation techniques (e.g., horizontal flipping, rotation) to improve generalization.

2. Transfer Learning:
- Pre-trained weights from ImageNet (MobileNet) and CheXNet (DenseNet) were used to initialize models.
- Enabled faster convergence and better generalization by leveraging previously learned features.

3. Regularization:
- Dropout layers (rates of 0.3–0.5) were added to prevent overfitting.
- L2 regularization applied to the dense layers to penalize large weights.

4. Evaluation Metrics:
- Binary Accuracy: Fraction of correctly classified labels in the multi-label task.
- Loss: Binary cross-entropy to handle independent label probabilities.
- AUC-ROC: Measures the model's ability to rank predictions correctly, crucial for multi-label classification.

## Conclusion
This project demonstrates the importance of transfer learning and lightweight architectures for thorax disease prediction. MobileNet with ImageNet weights emerged as the most balanced model, achieving high accuracy, low loss, and computational efficiency. DenseNet with CheXNet weights excelled in distinguishing diseases but required more resources and was slightly less effective overall. Future work may include experimenting with other medical-specific pre-trained models and optimizing data augmentation strategies.

## Acknowledgment
Special thanks to the National Institutes of Health (NIH) for providing the Chest X-rays dataset and the open-source community for pre-trained models like ImageNet and CheXNet, which made this project possible.
