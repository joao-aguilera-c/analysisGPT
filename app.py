from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
from time import sleep
import json
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import tiktoken

app = Flask(__name__)

MODEL = {
    'INTRO': "gpt-3.5-turbo",
    'ANALYSIS': "gpt-4",
    'CONCLUSION': "gpt-3.5-turbo"
}

DEBUG = {
    'INTRO': True,
    'ANALYSIS': False,
    'CONCLUSION': False
}

def get_data_description(df):
    # Create a prompt for the GPT system
    prompt = """You are AnalysisGPT, a large language model trained by OpenAI. Your mission is to describe a csv file as a data analyst would.
You will not receive the full CSV file. You will only receive a sample of 20 rows. But you will receive the number of rows and columns after the sample.
Don't mention you are analyzing a sample, nor talk anything about the number of rows or columns.

You can generalize the data and do your comments as if you were analyzing the full dataset.

Respond using HTML. You can use <h3> to <h6> for headers, <p> for paragraphs, <ul> and <li> for lists, <b> for bold, <i> for italics.

Your only task is to describe the data. You must awnser the following questions:
- Can you summarize the data in the CSV file in natural language? For example, can you describe the dataset and its contents in a few sentences?
- Can you talk what you think about this dataset? what it is for, and what conclusions you can make?

Your initial description will be later used by the model, to generate a json with graphs and tables.
"""

    # Get the number of rows and columns in the dataframe
    num_rows, num_cols = df.shape


    # get aleatory sample of 20 rows
    df = df.sample(20)

    messages = [
            {"role": "system", "content": f"{prompt}\n\nShape of the data: {num_rows} rows and {num_cols} columns"},
            {"role": "user", "content": f"{df.to_csv(index=False)}"}
    ]

    print("Sending request to OpenAI API...")
    try:
        response = openai.ChatCompletion.create(
            model=MODEL['INTRO'],
            messages=messages
        )
    except Exception as e:
        print("Error while sending request to OpenAI API")
        print(e)
        traceback.print_exc()
        return {
            "messages": "Error while sending request to OpenAI API",
            "description": e
        }
    print("Respose received!")

    # Extract the response from the API
    description = response.choices[0].message.content.strip()

    
#     description = '''This CSV file contains data of 50,000 job applicants' details. The column names are First Name, Last Name, Email, Application Date, Country, YOE (Years of Experience), Seniority, Technology, Code Challenge Score, and Technical Interview Score. The data seems to be related to job applications, including the candidate's personal information, professional background, and performance scores in code challenges and technical interviews.

# Based on the given data, one could infer that this dataset is of candidate profiles who applied for a job at a company. The company may have screened candidates based on their application date, YOE, seniority in their field, preferred technology, and code challenge and technical interview scores. From this data, the company can analyze the candidates in the order of their preference and select the best candidate for the job role. It could also be used to study hiring patterns across various countries.

# Distribution graphs and statistical tests can be performed on the given dataset to understand the frequency of candidates for each technology, seniority and country-wise. Interrelationships between variables can also be studied to determine which factors are most likely to influence candidate selection, such as preferred technology, seniority, or country.'''

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
        "comment": "A comment, described in section 1",
        "graph_code": "Python code, explained in section 2",
        "graph_to_text_code": "Python code, explained in section 3"
    },
    "cell2": {
        "comment": "A comment, described in section 1",
        "graph_code": "Python code, explained in section 2",
        "graph_to_text_code": "Python code, explained in section 3"
    },
    "cell3": {
        "comment": "A comment, described in section 1",
        "graph_code": "Python code, explained in section 2",
        "graph_to_text_code": "Python code, explained in section 3"
    }
}

Section 1:
    Here you can introduce the reason why you want to generate this graph, and what you want to show with it.
    Be talkative, explain your thought. The reason for choosing the graph is more important than the description of it.
    DON'T PREDICT WHAT THE GRAPH WILL SHOW.
    Remember, as a Data Analyst, you are constructing a narrative. You are telling a story with the data.

Section 2:
    This is a python code for generating the image. You can generate it using matplotlib or seaborn. Just import it.
    Make it look nice!
    In the end, you will save the plot as a image in the directory static/graph{i}.png, where i is the number of the cell.
    Always save with bbox_inches = 'tight'. Dont use rotation 45, use rotation = 90, ha = 'right'.
    YOU MUST CREATE ONLY 3 GRAPHS, ONE FOR EACH CELL."

Section 3:
    This is a python code for generating tables that will have the same information as the graphs. These tables is to give you more information about the graphs.
    With them, you will have a better understanding of the data in the graphs.
    In another moment, you will receive the outputs of this code, and you will use it to write a conclusion about the graphs you generated.
    - NEVER save a csv containing whole colums. The goal of this csv is to give you more information about the graph, not to give you the whole data.
    - Do not print the tables, save them as csv files in the directory data/graph_to_text{i}.csv, where i is the number of the cell. Use pd.to_csv().
    - When genarating the csv, use index = False
    - You can only generate one csv file for each graph. Choose the csv wisely as it will be your only source of information to write the conclusion.


Also you will receive some additional information about the csv file, such as the number of unique values for each column. Use this data to generate readable graphs. I.e. if a column has more than 10 unique values, you should not generate a barplot, but a histogram. Also, if a column has more than 10 unique values, you should not put this column in the any axis, as it will be unreadable.

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
            model=MODEL['ANALYSIS'],
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
            print("Error: ", e, " in the following json string:", json_string)
            messages.append({"role": "assistant", "content": json_string})
            messages.append({"role": "user", "content": str(e) + "Fix the error and return ONLY the json string."})

            print("Sending request to OpenAI API...")
            print(messages)
            response = openai.ChatCompletion.create(
                model=MODEL['ANALYSIS'],
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
                traceback.print_exc()
                error_message.append([f"cell{i}", f"Error: {e}"])

            # if there is an error, send the error message back to ChatGPT
            if error_message:
                messages.append({"role": "assistant", "content": json_string})

                error_message_content = "here is a list of errors:\n"
                for error in error_message:
                    error_message_content += f"{error[0]}: {error[1]}\n"
                error_message_content += "Please fix the errors and send the whole json(with the 3 cells) again."

                messages.append({"role": "user", "content": str(error_message)})
                print("There was an error, sending the error message back to ChatGPT...")
                print(messages[-1])

    # for now we will just return the json string
    return json_dict


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_conclusion(analysis_report, df):
    # Initialize an empty dictionary to store the conclusions
    conclusions = {}

    # Initialize an empty string to store the user prompt
    user_prompt = ""
    messages = analysis_report["messages"]

    messages[0] = {"role": "system", "content": "You, as AnalysisGPT, have to describe a CSV file like a data analyst, but you will only receive a sample of 20 rows, even though you'll know the shape of the full dataset. You can't mention that you're analyzing a sample. You must summarize the data in the CSV file in natural language and talk about what you think about the dataset, what it is for, and so on. You must use HTML (starting in <h3>) to format your response, and your initial description will be used to generate a JSON with graphs and tables."}
    messages.append({"role": "assistant", "content": analysis_report["description"]})
    messages.append({"role": "user", "content": "Now genarate a JSON containing 3 cells, each one with a graph_code, that will generate a graph image based on the df. Also a graph_to_text_code that will generate csvs with the same information as the graphs. The csvs will be used by you to generate conclusions regarding the whole dataset."})
    messages.append({"role": "assistant", "content": str(analysis_report["analysis"])})
    # Iterate through the cells in the analysis_report
    for cell_key, cell_value in analysis_report["analysis"].items():
        # Get the graph_to_text_code for the current cell
        graph_to_text_code = cell_value["graph_to_text_code"]

        # Run the graph_to_text_code
        try:
            print(f"Running {cell_key} graph_to_text_code:")
            print(graph_to_text_code)
            exec(graph_to_text_code)
        except Exception as e:
            print(f"Error in {cell_key} graph_to_text_code: {e}")
            traceback.print_exc()
            conclusions[cell_key] = f"Error in {cell_key} graph_to_text_code: {e}"
            continue

        # Read the generated CSV file
        csv_path = f"data/graph_to_text{cell_key[-1]}.csv"
        try:
            graph_to_text_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            traceback.print_exc()
            conclusions[cell_key] = f"Error reading {csv_path}: {e}"
            continue

        # Append the CSV content to the user prompt
        user_prompt += f"\n\n{cell_key}:\n{graph_to_text_df.to_csv(index=False)}"

    # Create a prompt for the GPT system
    prompt = f"""You're still AnalysisGPT. Your mission now is to analyze the data generated by graph_to_text_code provide a conclusion based on the data.

{user_prompt}

Please provide a conclusion in natural language, discussing the insights and patterns you observe in the data. DO NOT CREATE A CONCLUSION FOR EACH CELL, but rather a conclusion for the whole dataset.
If you want to format the text for better readability, you can use html tags."""

    messages.append({"role": "user", "content": prompt})

    # Send the request to the OpenAI API
    print("Sending request to OpenAI API...")
    print("Number of tokens in prompt:", num_tokens_from_string(prompt, "gpt-3.5-turbo"))
    try:
        response = openai.ChatCompletion.create(
            model=MODEL['CONCLUSION'],
            messages=messages
        )
    except Exception as e:
        print(f"Error while sending request to OpenAI API: {e}")
        traceback.print_exc()
        return {"error": f"Error while sending request to OpenAI API: {e}"}
    print("Respose received!")
    # Extract the response from the API
    conclusion = response.choices[0].message.content.strip()

    # Store the conclusion in the conclusions dictionary
    conclusions["conclusion"] = conclusion

    return conclusions

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')


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
    if DEBUG['INTRO']:
        sleep(3)
        # read 'debug/results.json' file
        with open('debug/results.json', 'r') as f:
            results = json.load(f)
        
        print(type(results))
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
    print("Generating additional results...")
    if DEBUG['ANALYSIS']:
        # read 'debug/additional_results.json' file
        sleep(5)
        with open('debug/additional_results.json', 'r') as f:
            additional_results = json.load(f)
    else:
        additional_results = get_additional_results(results_json, df)
        with open('debug/additional_results.json', 'w') as f:
            json.dump(additional_results, f)
        print("Additional results generated!")

    # Generate URLs for the graph images
    for i, cell in enumerate(additional_results):
        additional_results[cell]['graph_url'] = f"/static/graph{i+1}.png"

    return jsonify(additional_results)

@app.route('/conclusion', methods=['POST'])
def conclusion():
    results_json = request.get_json()

    df = pd.read_parquet("data/data.parquet")
    print("Generating conclusion...")
    if DEBUG['CONCLUSION']:
        # read 'debug/conclusion.json' file
        with open('debug/conclusion.json', 'r') as f:
            conclusion = json.load(f)
        conclusion = conclusion["conclusion"]
    else:
        conclusion = get_conclusion(results_json, df)
        with open('debug/conclusion.json', 'w') as f:
            json.dump({"conclusion":conclusion}, f)

    print("Conclusion generated!")
   
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