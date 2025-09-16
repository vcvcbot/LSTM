# LSTM IMDB Sentiment Analysis Project

## Project Introduction

This project aims to perform sentiment analysis on the IMDB movie review dataset using a Long Short-Term Memory (LSTM) network. Through a deep learning model, we classify reviews into positive or negative sentiments, demonstrating the application of Natural Language Processing (NLP) in text classification tasks. The project covers the complete pipeline from data loading, preprocessing, model building, training, to evaluation, and provides a detailed analysis of model performance.

## Features

*   **Data Exploration and Visualization**: Initial exploration of the IMDB dataset, including review length distribution, word frequency analysis, and visualization through charts.
*   **Text Preprocessing**: Implementation of various text cleaning techniques, including HTML tag removal, punctuation handling, stop word removal, lemmatization, etc., to optimize model input.
*   **Word Embedding**: Utilization of `Tokenizer` for vocabulary building and sequence padding, converting text into numerical representations.
*   **LSTM Model Construction**: Design and implementation of an LSTM network based on Keras to capture long-term dependencies in text.
*   **Model Training and Optimization**: Employment of callbacks such as `EarlyStopping` and `ReduceLROnPlateau` to effectively prevent overfitting and optimize the training process.
*   **Model Evaluation**: Comprehensive assessment of model performance through metrics like accuracy, loss, confusion matrix, classification report, and ROC curves.
*   **Error Analysis**: In-depth analysis of misclassified samples, identifying False Positives and False Negatives to understand model limitations.

## Dataset

This project uses the [IMDB Movie Review Sentiment Analysis Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset contains 50,000 movie reviews, each labeled as either positive or negative sentiment.

## Environment Setup

### Prerequisites

*   Python 3.x
*   pip (Python package installer)

### Installation

Please run the following command to install all necessary Python packages:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

## Usage

1.  **Clone the repository**:

    ```bash
git clone <GitHub URL of the project>
cd LSTM_IMDB_Sentiment_Analysis
    ```

2.  **Download the dataset**:

    Download the `IMDB Dataset.csv` file from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project's root directory, or update the `data_path` variable in the Jupyter Notebook to point to the dataset's location.

3.  **Run the Jupyter Notebook**:

    ```bash
jupyter notebook LSTM_IMDB.ipynb
    ```

    Within the Jupyter environment, you can execute each cell step-by-step to observe the results of data processing, model training, and evaluation.

## Project Structure

```
. 
├── LSTM_IMDB.ipynb         # Jupyter Notebook containing all code and analysis
├── IMDB Dataset.csv        # IMDB movie review dataset (needs manual download)
└── README_en.md            # This README file
```

## Results

The model achieved a validation accuracy of approximately **83.37%** on the test set. Detailed training history, evaluation metrics (such as precision, recall, F1-score), and confusion matrices can all be found within the Jupyter Notebook.

### Training Process Visualization

The Jupyter Notebook includes charts for training accuracy, validation accuracy, training loss, and validation loss, as well as learning rate changes and overfitting analysis. These visualizations help in understanding the model's training dynamics and performance.

### Error Analysis

Error analysis was also performed, showcasing samples that the model most frequently misclassified as False Positives and False Negatives. This is crucial for identifying areas for future model improvement.

## Conclusion

This project successfully demonstrates how to use an LSTM network for sentiment analysis on IMDB movie reviews. Although the model achieved good performance, the error analysis reveals potential for future improvements, such as more sophisticated text representation methods or deeper model architectures.


