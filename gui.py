from flask import Flask, render_template, request , url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




# Same logic in the model building file 
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load the pre-trained LSTM model
loaded_lstm_model = load_model("my_lstm_model")

# Load the TextVectorization layer
encoder = tf.keras.layers.TextVectorization(max_tokens=1400)
# Adapt the TextVectorization layer with a dummy data
encoder.adapt(tf.constant(["dummy text"]))

# Function to preprocess a single title
def preprocess_title(text, words_to_keep_unchanged=None):
    if words_to_keep_unchanged is None:
        words_to_keep_unchanged = ["us"]
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Tokenize the input text into words
        words = word_tokenize(text)
        # Remove stop words, make words lowercase, and apply lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [
            word if word.lower() in words_to_keep_unchanged else lemmatizer.lemmatize(word.lower())
            for word in words
            if word.lower() not in stop_words and word.isalpha()
        ]
        lemmatized_text = ' '.join(lemmatized_words)
        return lemmatized_text
    else:
        return text

# Example user input function using the loaded model
def get_predictions(user_input, model, encoder):
    if isinstance(user_input, str):
        user_input = [user_input]  # Convert single title to a list for consistency

    # Preprocess user input
    preprocessed_titles = [preprocess_title(title) for title in user_input]

    # Tokenize and pad the preprocessed titles using the same encoder used during training
    encoded_user_input = encoder(preprocessed_titles).numpy()

    # Ensure that the input to the TextVectorization layer is a 1D array of strings
    encoded_user_input_1d = np.array(preprocessed_titles).reshape((-1, 1))

    # Make predictions
    predictions = model.predict(encoded_user_input_1d)

    # Assuming your model is a multi-class classification model
    predicted_labels = np.argmax(predictions, axis=1)

    # Return the predictions
    return predicted_labels

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Inside /predict endpoint")
    
    if request.method == 'POST':
        user_title = request.form['title']
        print(f"User Title: {user_title}")

        # Add your prediction logic here using the loaded LSTM model and encoder
        predictions = get_predictions(user_title, loaded_lstm_model, encoder)
        predicted_label = predictions[0]

        # Map predicted class to text label
        class_to_label_mapping = {
            0: "Media and Public Opinion",
            1: "Politics and Diplomacy",
            2: "Violence and Humanitarian Crisis"
        }

        predicted_label_text = class_to_label_mapping.get(predicted_label, "Unknown")
        print(f"Predicted Label: {predicted_label_text}")

        return render_template('index.html', prediction=f"Predicted Label: {predicted_label_text}")

    # Handle GET requests, if needed
    print("Returning template for GET request")
    return render_template('index.html', prediction=None)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)






