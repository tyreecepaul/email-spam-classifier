"""
Email/SMS Spam Classifier using Streamlit
A machine learning application to classify messages as spam or not spam.
"""

import os
import pickle
import string
from pathlib import Path
from typing import List, Optional

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class TextPreprocessor:
    """Handles text preprocessing for spam classification."""
    
    def __init__(self, download_dir: str = "./nltk_data"):
        """
        Initialize the text preprocessor.
        
        Args:
            download_dir: Directory to store NLTK data
        """
        self.download_dir = download_dir
        self._setup_nltk()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
    
    def _setup_nltk(self) -> None:
        """Setup NLTK data directory and download required resources."""
        os.makedirs(self.download_dir, exist_ok=True)
        nltk.download('stopwords', download_dir=self.download_dir, quiet=True)
        nltk.data.path.append(self.download_dir)
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize and clean input text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of cleaned and stemmed tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove punctuation and filter stop words
        cleaned_words = [
            word.strip(string.punctuation)
            for word in words
        ]
        
        # Stem words and remove empty strings and stop words
        return [
            self.stemmer.stem(word)
            for word in cleaned_words
            if word and word not in self.stop_words
        ]
    
    def transform_text(self, text: str) -> str:
        """
        Transform text by tokenizing, cleaning, and rejoining.
        
        Args:
            text: Input text to transform
            
        Returns:
            Cleaned and processed text
        """
        tokens = self.tokenize_and_clean(text)
        return " ".join(tokens)


class SpamClassifier:
    """Spam classification model wrapper."""
    
    def __init__(self, vectorizer_path: str, model_path: str):
        """
        Initialize the spam classifier.
        
        Args:
            vectorizer_path: Path to the TF-IDF vectorizer pickle file
            model_path: Path to the trained model pickle file
        """
        self.vectorizer = self._load_pickle(vectorizer_path)
        self.model = self._load_pickle(model_path)
        self.preprocessor = TextPreprocessor()
    
    @staticmethod
    def _load_pickle(file_path: str):
        """
        Load a pickle file safely.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Loaded object from pickle file
            
        Raises:
            FileNotFoundError: If the pickle file doesn't exist
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    
    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict if a message is spam or not.
        
        Args:
            text: Input message to classify
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        if not text.strip():
            return "Not Spam", 0.5
        
        # Preprocess the text
        transformed_text = self.preprocessor.transform_text(text)
        
        # Vectorize the text
        vector_input = self.vectorizer.transform([transformed_text])
        
        # Make prediction
        prediction = self.model.predict(vector_input)[0]
        
        # Get prediction probability for confidence
        try:
            probabilities = self.model.predict_proba(vector_input)[0]
            confidence = max(probabilities)
        except AttributeError:
            # Model doesn't support predict_proba
            confidence = 1.0
        
        label = "Spam" if prediction == 1 else "Not Spam"
        return label, confidence


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Spam Classifier",
        page_icon="📧",
        layout="centered"
    )
    
    st.title("📧 Email/SMS Spam Classifier")
    st.markdown("Enter a message below to check if it's spam or not.")
    
    # Initialize the classifier
    try:
        classifier = SpamClassifier('vectorizer.pkl', 'model.pkl')
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()
    
    # User input
    input_message = st.text_area(
        "Enter your message:",
        placeholder="Type your email or SMS message here...",
        height=100
    )
    
    if st.button("Classify Message", type="primary"):
        if input_message.strip():
            with st.spinner("Analyzing message..."):
                prediction, confidence = classifier.predict(input_message)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == "Spam":
                    st.error(f"🚨 **{prediction}**")
                else:
                    st.success(f"✅ **{prediction}**")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show processed text in expander
            with st.expander("View Processed Text"):
                processed_text = classifier.preprocessor.transform_text(input_message)
                st.code(processed_text if processed_text else "No meaningful words found")
        else:
            st.warning("Please enter a message to classify.")
    
    # Add some information
    with st.expander("ℹ️ About this classifier"):
        st.markdown("""
        This spam classifier uses:
        - **Text preprocessing**: Removes punctuation, stop words, and applies stemming
        - **TF-IDF vectorization**: Converts text to numerical features
        - **Machine learning model**: Trained to distinguish spam from legitimate messages
        
        The model analyzes the content and structure of your message to make predictions.
        """)


if __name__ == "__main__":
    main()