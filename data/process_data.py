# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Merges the message.csv and categories.csv sources together
    :param messages_filepath: The path of the messages.csv
    :param categories_filepath: The path of the categories.csv
    :return: A dataframe merged on the common ID field
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(categories, messages, on='id')


def clean_data(df):
    """
    Processes the categories column of the dataset and converts it to an integer
    :param df: A dataframe containing messages and categories
    :return: A dataframe with each dummy category field converted to an integer
    """
    categories = df['categories'].str.split(";", expand=True)
    category_colnames = categories.columns = categories.iloc[0].str[:-2]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'], axis=1)
    df_joined = pd.concat([df, categories], axis=1)
    return df_joined.drop_duplicates(subset=['message'])


def save_data(df, database_filename):
    """
    Updates the messages entity in the passed database
    :param df: The dataframe to write to the database
    :param database_filename: the location of the database to write to
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    Executes the functions to extract, transform, and load the dataset.
    :return: None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
