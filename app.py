from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
from time import sleep
import json
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_KEY")

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
    description = '''This CSV file contains data of 50,000 job applicants' details. The column names are First Name, Last Name, Email, Application Date, Country, YOE (Years of Experience), Seniority, Technology, Code Challenge Score, and Technical Interview Score. The data seems to be related to job applications, including the candidate's personal information, professional background, and performance scores in code challenges and technical interviews.

Based on the given data, one could infer that this dataset is of candidate profiles who applied for a job at a company. The company may have screened candidates based on their application date, YOE, seniority in their field, preferred technology, and code challenge and technical interview scores. From this data, the company can analyze the candidates in the order of their preference and select the best candidate for the job role. It could also be used to study hiring patterns across various countries.

Distribution graphs and statistical tests can be performed on the given dataset to understand the frequency of candidates for each technology, seniority and country-wise. Interrelationships between variables can also be studied to determine which factors are most likely to influence candidate selection, such as preferred technology, seniority, or country.'''

    # Return results, a dictionary with the messages and the description
    results = {
        "messages": messages,
        "description": description
    }
    return results


def get_additional_results(initial_results, df):
    """This funtion will return a json containing strings and gaph images, to be displayed in the page"""

    prompt = """You are AnalysisGPT, a large language model trained by OpenAI. Your mission is to behave like a genius Senior Data Analyst, and generate a json string that will contain 3 keys named cell1, cell2 and cell3
This json will be created based on a previous data analysis of a csv file. The data analysis will be done by the same model that is generating this json.
You will receive a sample of 20 rows of the csv file and a description of the csv, in natural language. 

The json will be structured as follows:

{
    "cell1": {
        "comment": "see content of section 1",
        "graph_code": "see content of section 2"
    },
    "cell2": {
        "comment": "see content of section 1",
        "graph_code": "see content of section 2"
    },
    "cell3": {
        "comment": "see content of section 1",
        "graph_code": "see content of section 2"
    }
} 

Section 1: 
    Here you can introduce the reason why you want to generate this graph, and what you want to show with it. 
    Be talkative, explain your thought. The reason for choosing the graph is more important than the description of it. 
    Remember, as a Data Analyst, you are constructing a narrative. You are telling a story with the data.
Section 2: 
    This is a python code for generating the image. You can generate it using matplotlib or seaborn. Just import it.
    Make it look nice! 
    In the end, you will save the plot as a image in the directory static/graph{i}.png. 
    Always save with bbox_inches = 'tight'. If you use rotation 45, use rotation = 45, ha = 'right'.
    YOU MUST CREATE ONLY 3 GRAPHS, ONE FOR EACH CELL."


Also you will receive some additional information about the csv file, such as the number of unique values for each column. Use this data to generate readable graphs.

On the code, you can name the dataframe containing the data as df, this variable will be available to you here in the server and you can use it to generate the graphs.

After your reply we will run the codes, and if there is any error, we will ask you to fix it.

ATTENTION: YOUR REPLIES MUST BE IN JSON FORMAT. NO INTRODUCTION TEXTS, NO COMMENTS, NO EXPLANATIONS. ONLY THE JSON STRING.

"""

    user_content = initial_results["messages"][-1]["content"]

    # generate a series of important dataframes to give the model more information
    unique_values = df.nunique()

    # add the description to the prompt
    user_content += "\n\n" + "Unique Values in each column:\n" + unique_values.to_string()

    user_content += "\n\n" + initial_results["description"]

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    error_message = True
    while error_message:
        print("Sending request to OpenAI API...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            presence_penalty=2
        )
        print("Respose received!")

        # Extract the response from the API
        json_string = response.choices[0].message.content.strip()

        try:
            # Convert the json string to a dictionary
            json_dict = json.loads(json_string)
        except Exception as e:
            # If there is an error, send the error message back to ChatGPT
            messages.append({"role": "assistant", "content": json_string})
            messages.append({"role": "user", "content": e})

            print("Sending request to OpenAI API...")
            print(messages)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1
            )
        
            # Extract the response from the API
            json_string = response.choices[0].message.content.strip()
            json_dict = json.loads(json_string)

        error_message = []
        for i in range(1,4):
            # get the graph code
            graph_code = json_dict[f"cell{i}"]["graph_code"]
            # run the code
            print("Running the graph code:")
            print(graph_code)

            # clear the plot
            plt.clf()
            # run the code
            try:
                exec(graph_code)
            except Exception as e:
                # turn e into a string that is json serializable
                e = str(e)
                print(f"Error: {e}")
                error_message.append([f"cell{i}", f"Error: {e}"])
    
            # if there is an error, send the error message back to ChatGPT
            if error_message:
                messages.append({"role": "assistant", "content": json_string})
            
                error_message_content = "here is a list of errors:\n"
                for error in error_message:
                    error_message_content += f"{error[0]}: {error[1]}\n"
                error_message_content += "Please fix the errors and send the json again."    
            
                messages.append({"role": "user", "content": error_message})
                print("There was an error, sending the error message back to ChatGPT...")
                print(messages[-1])

    # for now we will just return the json string
    return json_dict


# Define the home page route
@app.route('/')
def home():
    results = {
        'messages': [
            {'role': '', 'content': ''},
            {'role': '', 'content': ''}
        ],
        'description': ''
    }
    return render_template('index.html', results=results)


def get_delimiter(file):
    """This function will return the delimiter of the file"""
    # open the file
    with open(file, 'r') as f:
        # read the first line
        line = f.readline()
        # get the number of commas
        commas = line.count(',')
        # get the number of semicolons
        semicolons = line.count(';')
        # get the number of tabs
        tabs = line.count('\t')
        # get the number of spaces
        spaces = line.count(' ')
        # get the number of pipes
        pipes = line.count('|')

    # create a dictionary with the number of each delimiter
    delimiters = {
        ',': commas,
        ';': semicolons,
        '\t': tabs,
        ' ': spaces,
        '|': pipes
    }

    # get the delimiter with the highest number
    delimiter = max(delimiters, key=delimiters.get)

    return delimiter


# Define the route for uploading a CSV file
@app.route('/upload', methods=['POST'])
def upload():
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
    results = get_data_description(df)

    
    # Return to the main page with the results
    return render_template('index.html', results=results)


# # Define the route for additional results
@app.route('/additional_results', methods=['POST'])
def additional_results():
    results_json = request.get_json()

    df = pd.read_parquet("data/data.parquet")
    print("Generating additional results...")
    additional_results = get_additional_results(results_json, df)
 
    print("Additional results generated!")

    # Generate URLs for the graph images
    for i, cell in enumerate(additional_results):
        additional_results[cell]['graph_url'] = f"/static/graph{i+1}.png"

    return jsonify(additional_results)


if __name__ == '__main__':
    # if doesn't exist, create data and static folders
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)