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
import pickle
import numpy as np


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='messages', con=engine)
    x = df['message']
    y = df.drop(['id', 'message', "original", 'genre'], axis=1)
    return x, y, y.columns

def tokenize(text):
    text = word_tokenize(text)
    tokens = []
    for word in text:
        word = WordNetLemmatizer().lemmatize(word).lower().strip()
        tokens.append(word)
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    return pipeline


def evaluate_model(model, x_test, y_test, category_names):
    y_pred = model.predict(x_test)
    output_df = []
    for column in range(len(category_names)):
        print(category_names[column])
        print(classification_report(np.array(y_test)[:, column], y_pred[:, column]))
        output_df.append([category_names[column].replace("_", " "), classification_report(np.array(y_test)[:, column],
                                                                        y_pred[:, column], output_dict=True).get(
            'accuracy')])
    print("saving evaluation to ./model/eval.pkl")
    pd.DataFrame(output_df, columns=['Category', 'Accuracy']).to_pickle("./models/eval.pkl")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
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
