# -AI-Powered-IoT-Botnet-Detection
AI-Powered IoT Botnet Detection

This project implements a machine learning and deep learning–based framework for detecting botnet attacks in IoT networks using network traffic data. The goal is to accurately distinguish benign IoT traffic from botnet-generated malicious traffic while minimizing false alarms and missed detections.


Project Objective

The rapid growth of IoT devices has increased vulnerability to botnet attacks such as Mirai and BASHLITE. This project aims to:

Detect IoT botnet attacks using supervised learning

Compare classical machine learning and deep learning models

Evaluate models using security-relevant metrics

Analyze accuracy vs. computational cost trade-offs


Models Implemented
Machine Learning

Support Vector Machine (SVM)

Decision Tree (DT)

Random Forest (RF)

Deep Learning

Convolutional Neural Network (CNN)

Recurrent Neural Network (RNN – LSTM)

Generative Adversarial Network (GAN) for data imbalance handling


 Dataset Overview

Dataset: NaiBot (IoT Botnet Dataset)

Traffic Types:

Benign IoT traffic

Botnet traffic (Mirai, BASHLITE variants)

Features:

115 numerical network traffic features

Extracted across multiple time windows

Labels:

0 → Benign

1 → Attack


 Dataset files are not included in this repository due to size and licensing constraints.


 Evaluation Metrics

The models are evaluated using:

Accuracy

Precision

Recall

F1-score

False Alarm Rate (FAR)

Missed Detection Rate (MDR)

Training and inference time

These metrics ensure suitability for real-world IoT security systems, where false alarms are costly.


Key Findings

Random Forest achieved the best overall performance with near-perfect accuracy

Decision Tree offered the fastest training and inference time

CNN performed competitively, showing strong feature-learning capability

SVM and GAN showed poor performance for this dataset and task


 Technologies Used

Python

Scikit-learn

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn



Author
Afsana Sharmin
PhD Researcher | AI, Machine Learning, Deep Learning, Cybersecurity