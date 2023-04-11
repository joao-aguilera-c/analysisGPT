from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
from time import sleep
import json
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from utils import (
    get_data_description, get_additional_results, get_conclusion, 
    get_delimiter
)

from app_config import Config

# configure logging format to have: time, file, line, message
logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)

app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for uploading a CSV file
@app.route('/upload', methods=['POST'])
def upload():

    # Get OpenAI API key
    openai.api_key = request.form['openai-api-key']

    # Get the uploaded file
    file = request.files['file']

    file_dir = f'data/{file.filename}'

    #save the file to the server
    file.save(file_dir)

    # get the file delimiter
    delimiter = get_delimiter(file_dir)

    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(file_dir, delimiter=delimiter)

    # save the dataframe to a parquet file
    df.to_parquet("data/data.parquet")

    # sleep(3)
    # Process the data to create a description of the data using chatGPT
    if Config.get('BYPASS_MODEL').get('INTRO'):
        sleep(3)
        # read 'debug/results.json' file
        with open('debug/results.json', 'r') as f:
            results = json.load(f)
        
        logging.info(type(results))
    else: 
        results = get_data_description(df)
        # save results to debugging purposes at debug folder (results is a dictionary)
        with open('debug/results.json', 'w') as f:
            json.dump(results, f)

    # Return to the main page with the results
    return jsonify(results)



# # Define the route for additional results
@app.route('/additional_results', methods=['POST'])
def additional_results():
    results_json = request.get_json()

    df = pd.read_parquet("data/data.parquet")
    logging.info("Generating additional results...")
    if Config.get('BYPASS_MODEL').get('ANALYSIS'):
        # read 'debug/additional_results.json' file
        sleep(5)
        with open('debug/additional_results.json', 'r') as f:
            additional_results = json.load(f)
    else:
        additional_results = get_additional_results(results_json, df)
        with open('debug/additional_results.json', 'w') as f:
            json.dump(additional_results, f)
        logging.info("Additional results generated!")

    # Generate URLs for the graph images
    for i, cell in enumerate(additional_results):
        additional_results[cell]['graph_url'] = f"/static/graph{i+1}.png"

    return jsonify(additional_results)


@app.route('/conclusion', methods=['POST'])
def conclusion():
    results_json = request.get_json()

    df = pd.read_parquet("data/data.parquet")
    logging.info("Generating conclusion...")
    if Config.get('BYPASS_MODEL').get('CONCLUSION'):
        # read 'debug/conclusion.json' file
        with open('debug/conclusion.json', 'r') as f:
            conclusion = json.load(f)
        conclusion = conclusion["conclusion"]
    else:
        conclusion = get_conclusion(results_json, df)
        with open('debug/conclusion.json', 'w') as f:
            json.dump({"conclusion":conclusion}, f)

    logging.info("Conclusion generated!")
   
    return jsonify(conclusion)


if __name__ == '__main__':
    # if doesn't exist, create data and static folders
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('debug'):
        os.makedirs('debug')

    app.run(debug=True)