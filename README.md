# text_classification_project_palestinian_occupation

## Description:
The Palestinian occupation Title Classifier is a machine learning project designed to classify titles related to the Palestinian occupation into 3 distinct categories. Leveraging Natural Language Processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network model, the system predicts the category of a given title among predefined classes : Media and Public Opinion/Politics and Diplomacy/Violence and Humanitarian Crisis. Trained on a dataset of titles specifically curated from subreddits related to the Palestinian occupation. Users can input titles and receive predictions instantly, aiding in the analysis and understanding of media discourse surrounding the Palestinian occupation.

![workflow (1)](https://github.com/AmmariHiba/text_classification_project_palestinian_occupation/assets/121461519/6629a8ff-f3b8-491c-9bd6-905bf7c84c1e)

## Key Features:
- Utilizes a pre-trained LSTM model for text classification.
- Implements text preprocessing techniques including tokenization, lemmatization, and stop words removal.
- Provides a web-based user interface for users to input titles and receive predictions on their topics.
- Developed with Python, Flask, TensorFlow and NLTK.
- Easy to set up and run for testing purposes.

## How To run :

### For Developers Interested in Exploring the Work:

- Data Collection: Refer to "Data Collection.ipynb" to understand how data was collected using the Reddit API.
- Labeling: Explore "labeling.ipynb" to learn about the process of labeling the collected data.
- Exploratory Data Analysis and Model Building: Dive into "EDA , Text processing and Models Training .ipynb" to analyze the data through exploratory data analysis (EDA), visualize graphs, and build classification models. Three models were developed: MultinomialNB, RandomForest, and LSTM. LSTM exhibited the best performance and was subsequently deployed.

### Running the GUI Application:

To run the graphical user interface (GUI) application for title prediction, ensure you have Python 3.9.12 installed along with the necessary libraries.

Load the Model: Place gui.py, the my_lstm_model folder, the templates folder, and the static folder in the same location. The my_lstm_model folder contains the pre-trained LSTM model, while the templates and static folders contain HTML and CSS files, respectively.

Running with Flask: Execute gui.py. This script leverages the Flask framework to create a web server. Flask will automatically load the LSTM model. Once the Flask server is up and running, navigate to the provided URL (usually http://127.0.0.1:5000/) in your web browser to access the GUI interface.

The GUI interface allows users to input titles and receive predictions on their topics based on the trained LSTM model.

![Screenshot 2024-02-06 at 17 24 41](https://github.com/AmmariHiba/text_classification_project_palestinian_occupation/assets/121461519/95cd982a-6ea9-41b7-bb86-1dc364befce1)



