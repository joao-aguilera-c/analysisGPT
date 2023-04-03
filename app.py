from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
from time import sleep
import json

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = "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"

def get_data_description(df):
    # Create a prompt for the GPT system
    prompt = """You are AnalysisGPT, a large language model trained by OpenAI. Your mission is to describe a csv file as a data analyst would. 
You will not receive the full CSV file. You will only receive a sample of 20 rows. But you will receive the number of rows and columns after the sample.
Don't mention you are analyzing a sample, nor talk anything about the number of rows or columns. 

You can generalize the data and do your comments as if you were analyzing the full dataset.

Respond using markdown.
Your only task is to describe the data. You must awnser the following questions:
- Can you summarize the data in the CSV file in natural language? For example, can you describe the dataset and its contents in a few sentences?
- Can you talk what you think about this dataset? what it is for, and what conclusions you can make?

Also, as you're a data analyst, talk about relevant interpretation you found regarding the data. For example, you can talk about the distribution of the data, the correlation between variables, etc. 
"""

    # Get the number of rows and columns in the dataframe
    num_rows, num_cols = df.shape


    # get aleatory sample of 20 rows
    df = df.sample(20)

    messages = [
            {"role": "system", "content": f"{prompt}\n\nShape of the data: {num_rows} rows and {num_cols} columns"},
            {"role": "user", "content": f"{df.to_csv(index=False)}"}
    ]

    # print("Sending request to OpenAI API...")
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages
    # )
    # print("Respose received!")

    # # Extract the response from the API
    # description = response.choices[0].message.content.strip()
    # # Add the number of rows and columns to the description
    # # description += f"\n\nThe CSV file has {num_rows} rows and {num_cols} columns."
    description = """This CSV file contains data of 50,000 job applicants' details. The column names are First Name, Last Name, Email, Application Date, Country, YOE (Years of Experience), Seniority, Technology, Code Challenge Score, and Technical Interview Score. The data seems to be related to job applications, including the candidate's personal information, professional background, and performance scores in code challenges and technical interviews.

Based on the given data, one could infer that this dataset is of candidate profiles who applied for a job at a company. The company may have screened candidates based on their application date, YOE, seniority in their field, preferred technology, and code challenge and technical interview scores. From this data, the company can analyze the candidates in the order of their preference and select the best candidate for the job role. It could also be used to study hiring patterns across various countries.

Distribution graphs and statistical tests can be performed on the given dataset to understand the frequency of candidates for each technology, seniority and country-wise. Interrelationships between variables can also be studied to determine which factors are most likely to influence candidate selection, such as preferred technology, seniority, or country."""

    # Return results, a dictionary with the messages and the description
    return {
        "messages": messages,
        "description": description
    }

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for uploading a CSV file
@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    file = request.files['file']

    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(file)

    # sleep(3)
    # Process the data to create a description of the data using chatGPT
    results = get_data_description(df)

    
    # Return to the main page with the results
    return render_template('index.html', results=results)


# # Define the route for additional results
@app.route('/additional_results', methods=['POST'])
def additional_results():
    results_json = request.get_json()
    # analysisIntroduction = data['analysisIntroduction']
    results = results_json.get('results')
    # convert string json to dict
    results = json.dumps(results)
    results = json.loads(results)
    print("Generating additional results...")
    additional_results = "Additional results"
    print("Additional results generated!")
    # Return the additional results
    sleep(3)
    return jsonify(additional_results)

if __name__ == '__main__':
    app.run(debug=True)