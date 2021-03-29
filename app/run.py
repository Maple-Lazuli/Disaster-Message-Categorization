import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Figure, Table
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

def generate_df_sums(df):
    dataframe = df.drop(['id', 'message', "original", 'genre'], axis=1)
    return pd.DataFrame({'Category': dataframe.sum(axis=0).index, 'Occurrences': dataframe.sum(axis=0).values})


# load model
model = joblib.load("../models/classifier.pkl")

# Load Accuracy Data Frame
eval_df = pd.read_pickle("../models/eval.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    cate_sums = generate_df_sums(df)
    dataframe = df.drop(['id'], axis=1)
    dataframe = dataframe[0:3]
    table = Figure([Table(
        header=dict(
            values=dataframe.columns[1:11],
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=[dataframe[k].tolist() for k in dataframe.columns[1:11]],
            align="left")
    )
    ])

    table.update_layout(
        autosize=True,
    )
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
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
        },
        {
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
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    tableJSON = json.dumps(table, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', ids=ids, graphJSON=graphJSON, tableJSON=tableJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
