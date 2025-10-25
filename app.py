# **Improvement 8: Streamlit UI Enhancement**
import streamlit as st
import pandas as pd
import altair as alt # For visualization
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Define the correct paths to the saved files in Google Drive ---
VECTORIZER_PATH = '/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'
MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/sentiment_model.h5'
TOKENIZER_PATH = '/content/drive/MyDrive/Colab Notebooks/lstm_tokenizer.pickle' # Assuming you saved this

# --- Load the saved model and vectorizer/tokenizer ---
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    st.success("TF-IDF vectorizer loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: {VECTORIZER_PATH} not found. Make sure you have saved it to this path.")
    vectorizer = None # Set to None to prevent errors later

try:
    # Custom objects might be needed if you used custom layers/functions
    model = load_model(MODEL_PATH)
    st.success("Neural network model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: {MODEL_PATH} not found. Make sure you have saved it to this path.")
    model = None # Set to None to prevent errors later
except Exception as e:
     st.error(f"Error loading Keras model: {e}")
     model = None

try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    st.success("Tokenizer loaded successfully!")
except FileNotFoundError:
     st.error(f"Error: {TOKENIZER_PATH} not found. Make sure you have saved it if using LSTM.")
     tokenizer = None
except Exception as e:
     st.error(f"Error loading tokenizer: {e}")
     tokenizer = None


# Define your label map
LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Define text cleaning function (should match the one used during training)
import re
def clean_text(text):
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    try:
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return ""

# Define padding length (should match the one used during training)
MAX_LEN = 100 # Or whatever MAX_LEN you used for padding

# --- Enhanced UI/UX and Prediction Logic ---

st.title("Sentiment Analysis Project Demo 📊")
st.markdown("Enter a customer review (text) below to instantly classify its sentiment (Negative, Neutral, or Positive).")
st.markdown("---")

user_input = st.text_area("Enter review here:", "")

if user_input and vectorizer is not None and model is not None and tokenizer is not None: # Proceed only if there is input and all models are loaded

    # 1. Prediction and Error Handling (Improvement 2: Streamlit Error Handling)
    try:
        # Preprocess the input text
        cleaned_input = clean_text(user_input)

        # If using TF-IDF (traditional models):
        # input_vector = vectorizer.transform([cleaned_input])
        # prediction_proba = model.predict_proba(input_vector)[0]

        # If using LSTM (deep learning model):
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction_proba = model.predict(padded_sequence)[0] # Predict returns probabilities for each class


        if prediction_proba is not None:
            # 2. Determine the predicted class
            predicted_class_index = prediction_proba.argmax()
            predicted_sentiment = LABEL_MAP[predicted_class_index]

            # 3. Display the primary prediction with clear formatting
            st.subheader("Analysis Result")
            if predicted_sentiment == 'Positive':
                st.success(f"**Predicted Sentiment:** {predicted_sentiment} 🎉 (Confidence: {prediction_proba.max():.2%})")
            elif predicted_sentiment == 'Negative':
                st.error(f"**Predicted Sentiment:** {predicted_sentiment} 😔 (Confidence: {prediction_proba.max():.2%})")
            else:
                st.warning(f"**Predicted Sentiment:** {predicted_sentiment} 🤔 (Confidence: {prediction_proba.max():.2%})")

            # 4. Visualize Prediction Probabilities
            st.subheader("Prediction Probability Distribution")

            proba_df = pd.DataFrame({
                'Sentiment': list(LABEL_MAP.values()),
                'Probability': prediction_proba
            }).sort_values(by='Probability', ascending=False)

            # Create a visually engaging bar chart
            chart = alt.Chart(proba_df).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='.0%')),
                y=alt.Y('Sentiment', sort='-x'),
                color=alt.condition(
                    alt.datum.Sentiment == predicted_sentiment,
                    alt.value('#28a745'),  # Green for predicted class
                    alt.value('steelblue')
                ),
                tooltip=['Sentiment', alt.Tooltip('Probability', format='.2%')]
            ).properties(
                title='Model Confidence Across Classes'
            )
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
