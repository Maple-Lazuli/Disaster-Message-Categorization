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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def generate_df_sums(input_df):
    dataframe = input_df.drop(['id', 'message', "original", 'genre'], axis=1)
    return pd.DataFrame({'Category': dataframe.sum(axis=0).index, 'Occurrences': dataframe.sum(axis=0).values})


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# Load Accuracy Data Frame
eval_df = pd.read_pickle("../models/eval.pkl")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/go')
def go():

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
    return render_template('sources.html')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
