# Fake News Detection

A machine learning-based application to detect and classify fake news articles using Natural Language Processing techniques.

## 📁 Project Structure

```
.
├── fake_news.py              # Main model training script
├── app_building.py           # Application/API implementation
├── datasets.zip              # Compressed dataset files
│   ├── sample_df.csv         # Sample training data
│   └── test_df.csv           # Test dataset
├── fake_news_model.pkl       # Trained classification model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer for text processing
└── README.md                 # Project documentation
```

## 🚀 Features

- Machine learning model for fake news classification
- TF-IDF vectorization for text feature extraction
- Pre-trained model ready for deployment
- Sample and test datasets included

## 📋 Prerequisites

```bash
python >= 3.7
pandas
scikit-learn
numpy
pickle
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Extract the datasets:
```bash
unzip datasets.zip
```

## 💻 Usage

### Training the Model

Run the training script to build and train the fake news detection model:

```bash
python fake_news.py
```

### Running the Application

Launch the application interface:

```bash
python app_building.py
```

## 📊 Dataset

The project includes two CSV files:
- **sample_df.csv**: Training dataset with labeled news articles (this is the sample dataset generated for uploading in the applucation developed using the streamlit)
- **test_df.csv**: Test dataset for model evaluation

## 🤖 Model Details

- **Model**: Saved as `fake_news_model.pkl`
- **Vectorizer**: TF-IDF vectorizer saved as `tfidf_vectorizer.pkl`
- The model uses text classification techniques to distinguish between real and fake news

## 🔍 How It Works

1. Text data is preprocessed and cleaned
2. TF-IDF vectorization converts text into numerical features
3. The trained model classifies news articles as real or fake
4. Results are displayed through the application interface

   
## Note
1.While uploading any dataset from the outside sources for testing the application please go through the rules mentioned in the application and ensure your dataset matches all those rules to ensure
  accurate predictions
2. I didnt not include the True.csv and Fake.csv datasets that i downloaded from the kaggle fake news dataset. While training the model please ensure that you combine the Subject and Title columns into
   a single column and then apply Tfifd vectorizer to ensure more accuracy.
## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👥 Authors

Your Name - Chivukula Sai Rithivik[rithivik-1512]

## 🙏 Acknowledgments

- Dataset sources
- Libraries and frameworks used
- Any other acknowledgments
