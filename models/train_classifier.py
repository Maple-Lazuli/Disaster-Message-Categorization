import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np


def load_data(database_filepath):
    """
    Load the data from the messages entity in the referenced database
    :param database_filepath: The path of the database to load data from
    :return: x - the feature to predict for, y - the features to use for prediction, y,columns - Names of the features
    used for prediction
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='messages', con=engine)
    x = df['message']
    y = df.drop(['id', 'message', "original", 'genre'], axis=1)
    return x, y, y.columns


def tokenize(text):
    """
    Takes the input string and splits it into a list. Then the function normalizes the text, removes extra whitespace,
    and lemmatizes the elements
    :param text: String containing natural language
    :return: a list of tokens
    """
    text = word_tokenize(text)
    tokens = []
    for word in text:
        word = WordNetLemmatizer().lemmatize(word).lower().strip()
        tokens.append(word)
    return tokens


def build_model():
    """
    Creates a scikit-learn pipeline that tokenizes, vectorizes, runs TF-IDF, then classifies whatever data is used when
    the pipeline is later ran.
    :return: a scikit-learn pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1,2), (1, 5)),
        'clf__estimator__n_neighbors': (5, 10, 15)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=6)
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluates the model, stores the results in a dataframe then writes the dataframe to the disk.
    :param model: A model that uses messages and predicts categories
    :param x_test: Messages that were separated from the model training set
    :param y_test: The correct features for prediction that were excluded from the training set
    :param category_names: A list of the feature names used for prediction
    :return: None
    """
    y_pred = model.predict(x_test)
    output_df = []
    for column in range(len(category_names)):
        print(category_names[column])
        print(classification_report(np.array(y_test)[:, column], y_pred[:, column]))
        output_df.append([category_names[column].replace("_", " "),
                          classification_report(np.array(y_test)[:, column], y_pred[:, column], output_dict=True).get(
            'accuracy')])
    print("saving evaluation to ./model/eval.pkl")
    pd.DataFrame(output_df, columns=['Category', 'Accuracy']).to_pickle("./models/eval.pkl")


def save_model(model, model_filepath):
    """
    Save machine learning model to the disk
    :param model: The model to write to the disk
    :param model_filepath: The directory to save in and name to save the model as
    :return: None
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Uses the functions in this file to load the data, split the data, build the model, train the model, evaluate the
    model and save the model.
    :return: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(x_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
