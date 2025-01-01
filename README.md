# Classification of Urdu News Articles

## Project Overview
This project focuses on building a machine learning pipeline to classify Urdu news articles into predefined categories:
- Entertainment
- Business
- Sports
- Science-Technology
- International

The project involves web scraping, preprocessing, exploratory data analysis (EDA), and implementing multiple machine learning models to identify the best-performing classifier.

## Motivation
Urdu, a widely spoken language, lacks sufficient content categorization tools. This project transforms unstructured news data into organized categories, forming the foundation for a personalized news system. This effort aims to simplify access to relevant content for Urdu-speaking users.

## Features
- **Data Collection**: Scraping Urdu news articles from various sources like Geo Urdu, ARY Urdu, etc.
- **Data Preprocessing**: Cleaning and EDA to prepare the dataset for machine learning.
- **Model Implementation**: Custom machine learning models for classification tasks.
- **Evaluation**: Comparing models using metrics to determine the best classifier.

## File Structure
- `Scraping.ipynb`: Script to scrape Urdu news articles.
- `EDA.ipynb`: Notebook for data cleaning and exploratory analysis.
- `models`: Folder containing scripts implementing classification models (Naive Bayes, Logistic Regression, Neural Networks)
- `Final_Report.pdf`: Detailed report of the methodology, results, and analysis.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed and the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `nltk`

### Evaluation
Models are compared based on the following metrics:
- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
