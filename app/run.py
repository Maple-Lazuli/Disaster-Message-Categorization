import json
import plotly
import pandas as pd
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Figure, Table
from sqlalchemy import create_engine
app = Flask(__name__)
"""
Summary of Routes:
    1. / and /index - Both return index.html and present the form to submit a message to query
    2. /go - return go.html which contains the form to submit a message as well as the previous classification results
    3. /overview - returns overview.html and provides information on the project, dataset, and model
    4. /sources - returns sources.html and provides references and acknowledgements.
"""


def tokenize(text):
    """
    Takes the input string and splits it into a list. Then the function normalizes the text, removes extra whitespace,
    and lemmatizes the elements
    :param text: String containing natural language
    :return: a list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def generate_df_sums(input_df):
    """
    Generates a dataframe containing a count of occurrences for each of the tags for the message.
    :param input_df: The dataset used for modeling to process
    :return: A dataframe containing the occurrence counts
    """
    dataframe = input_df.drop(['id', 'message', "original", 'genre'], axis=1)
    return pd.DataFrame({'Category': dataframe.sum(axis=0).index, 'Occurrences': dataframe.sum(axis=0).values})


# load data exported from process_data.py
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model exported from train_classifier.py
model = joblib.load("../models/classifier.pkl")

# Load accuracy dataframe from train_classifier.py
eval_df = pd.read_pickle("../models/eval.pkl")


@app.route('/')
@app.route('/index')
def index():
    """
    :return: A webpage that just contain the message submit form
    """
    return render_template('index.html')


@app.route('/go')
def go():
    """
    Takes the url argument from an HTTP GET message and processes the argument in the model
    :return: A webpage containing the results of the message processing in the model
    """
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/overview')
def overview():
    """
    Generates graphs and tables based on the model and dataset and then converts them to a JSON string
    :return: A webpage containing graphs and a table of the dataset and the machine learning model
    """
    cate_sums = generate_df_sums(df)
    dataframe = df.drop(['id', 'genre'], axis=1)
    dataframe = dataframe[0:3]
    table = Figure([Table(
        header=dict(
            values=dataframe.columns[0:11],
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=[dataframe[k].tolist() for k in dataframe.columns[0:11]],
            align="left")
    )
    ])
    table.update_layout(
        autosize=True,
    )
    counts_graph = {
            'data': [
                Bar(
                    x=cate_sums['Category'],
                    y=cate_sums['Occurrences']
                )
            ],
            'layout': {
                'title': '<b> Occurrences By Category </b>',
                'yaxis': {
                    'title': "<b> Occurrences % </b>"
                },
                'xaxis': {
                    'title': "<b> Category </b>"
                }
            }
        }
    accuracy_graph = {
            'data': [
                Bar(
                    x=eval_df['Category'],
                    y=eval_df['Accuracy']
                )
            ],

            'layout': {
                'title': '<b> Accuracy of Model Categories </b>',
                'yaxis': {
                    'title': "<b> Accuracy % </b>"
                },
                'xaxis': {
                    'title': "<b> Category </b>"
                }
            }
        }
    table_json = json.dumps(table, cls=plotly.utils.PlotlyJSONEncoder)
    counts_json = json.dumps(counts_graph, cls=plotly.utils.PlotlyJSONEncoder)
    accuracy_json = json.dumps(accuracy_graph, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('overview.html', countsJSON=counts_json, tableJSON=table_json, accuracyJSON=accuracy_json)


@app.route("/sources")
def sources():
    """
    :return: A webpage containing due credit for the creation of the project.
    """
    return render_template('sources.html')


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
