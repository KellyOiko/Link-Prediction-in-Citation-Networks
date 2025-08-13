# Link-Prediction-in-Citation-Networks

## Overview
This project tackles the challenge of predicting citation links between research papers in a large-scale citation network.  
The model combines NLP (textual similarity) and graph-based features to predict edges in the network, optimized to minimize log loss.

Given a pair of research papers, the task is to determine whether one cites the other.

---

## Problem Description
You are provided with:

- A citation network containing several thousand research papers
- Each paper’s abstract
- List of authors
- Citation relationships between papers

The goal is to predict whether a citation exists between two given papers.  
This task has applications in academic paper recommendation systems, social network analysis, and biological interaction prediction.

---

## Approach
The pipeline follows a standard supervised classification process:

**Feature Extraction**
- Textual features: abstract similarity (TF-IDF cosine similarity, embeddings)
- Author features: overlap or relatedness of authors
- Graph features: common neighbors, shortest path distance, Jaccard coefficient, preferential attachment, etc.

**Model Training**
- Positive examples: known citation links
- Negative examples: known non-links
- Train a probabilistic classifier to output the likelihood of a citation

**Prediction**
- For unseen paper pairs, output the predicted probability of a citation link

---

## Evaluation
The model is evaluated using **Log Loss**:

LogLoss = - (1/N) * Σ [ yᵢ * log(pᵢ) + (1 - yᵢ) * log(1 - pᵢ) ]


Where:
- **N** = number of samples  
- **yᵢ** = 1 if a citation exists, else 0  
- **pᵢ** = predicted probability of a citation  

Lower log loss indicates better performance.

---

## Modeling Details
The final model is a **LightGBM binary classifier** trained on the complete feature set, which includes:
- Graph-based features
- Textual similarity features
- Author-related features

---

## Training Procedure
- **Cross-Validation**: Used Stratified K-Fold Cross-Validation with n = 10 folds to maintain balanced label distribution across splits.
- **Metric**: Optimized for log loss.
- **Efficiency**: Incorporated early stopping and logging callbacks to improve training speed and monitor progress.
- **Model Storage**: Each fold's trained model was saved for potential reuse or ensembling.

---

## Prediction & Submission
- **Ensembling**: Combined predictions from all trained models by averaging their probabilities to achieve more stable and reliable results.
- **Output**: Final averaged predictions on the test set were saved as a `.csv` file, ready for submission.
