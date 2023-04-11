import openai
import pandas as pd
import json
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import tiktoken
from app_config import Config
import logging

logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_data_profile(df):
    # Create a profile report of the data
    from ydata_profiling import ProfileReport
    import itertools

    profile = ProfileReport(df, title="Profiling Report")
    # As a JSON string
    profile = profile.to_json()
    profile = json.loads(profile)
    keys = ['table', 'variables', 'alerts']
    profile = {key: profile[key] for key in keys}
    # rename 'variables' key to 'columns'
    profile['columns'] = profile.pop('variables')
    # on profile['columns']['value_counts_without_nan'] keep the first 10 values of each column
    columns_pop_keys = [
        'hashable', 'value_counts_without_nan', 'value_counts_index_sorted',
        'ordering', 'n', 'count', 'memory_size', 'imbalance',
        'first_rows', 'chi_squared', 'max_length', 'mean_length',
        'median_length', 'min_length', 'length_histogram', 'histogram_length',
        'n_characters_distinct', 'n_characters', 'character_counts', 
        'category_alias_values', 'block_alias_values', 'block_alias_counts',
        'n_block_alias', 'block_alias_char_counts', 'script_counts', 
        'n_scripts', 'script_char_counts', 'category_alias_counts',
        'n_category', 'category_alias_char_counts', 'word_counts',
        'n_negative', 'p_negative', 'n_infinite', 'n_zeros', 
        'std', 'variance', 'kurtosis', 'skewness', 'sum', 'mad',
        'chi_squared', 'range', 'iqr', 'cv', 'p_zeros', "p_infinite",
        'monotonic_increase', 'monotonic_decrease', 'monotonic_increase_strict',
        'monotonic_decrease_strict', 'monotonic', 'histogram'
    ]

    for column in profile['columns']:
        profile['columns'][column]['top3_value_counts_without_nan'] = dict(itertools.islice(profile['columns'][column]['value_counts_without_nan'].items(), 3))
        
        for key in columns_pop_keys:
            profile['columns'][column].pop(key, None)

    return profile


def get_data_description(df):
    # Create a prompt for the GPT system
    prompt = """You are AnalysisGPT, a large language model trained by OpenAI. Your mission is to describe a csv file as a data analyst would.
You will not receive CSV file, but a profile report of the data. You must use the information in the profile report to describe the data.
Don't mention you are analyzing a profile tho. Just talk about the data.

Respond using HTML. You can use <h3> to <h6> for headers.

Your only task is to describe the data. You must awnser the following questions:
- What is the Summary of the data?
- Can you give a brief description of the data?

Your initial description will be later used by the model, to generate a json with graphs and tables.
"""

    # Get data profile
    profile = get_data_profile(df)

    messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(profile)}
    ]

    logging.info("Sending request to OpenAI API...")
    try:
        response = call_gpt(messages, Config.get('MODEL').get('INTRO'))
    except Exception as e:
        logging.info("Error while sending request to OpenAI API")
        logging.info(e)
        traceback.print_exc()
        return {
            "messages": "Error while sending request to OpenAI API",
            "description": e
        }
    logging.info("Respose received!")

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

    prompt = """You are AnalysisGPT, a large language model trained by OpenAI. Your mission is to behave like a genius Senior Data Analyst, and generate a json string that will contain 3 keys named graph1, graph2 and graph3
This json will be created based on a previous data analysis of a csv file. The data analysis will be done by the same model that is generating this json.
You will receive a sample of 20 rows of the csv file and a description of the csv, in natural language.

The json will be structured as follows:

{
    "graph1": {
        "comment": "A comment, described in section 1",
        "graph_code": "Python code, explained in section 2",
        "graph_to_text_code": "Python code, explained in section 3"
    },
    "graph2": {
        "comment": "A comment, described in section 1",
        "graph_code": "Python code, explained in section 2",
        "graph_to_text_code": "Python code, explained in section 3"
    },
    "graph3": {
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
    In the end, you will save the plot as a image in the directory static/graph{i}.png, where i is the number of the graph.
    Always save with bbox_inches = 'tight'. Dont use rotation 45, use rotation = 90, ha = 'right'.
    YOU MUST CREATE ONLY 3 GRAPHS, ONE FOR EACH CELL."

Section 3:
    This is a python code for generating tables that will have the same information as the graphs. These tables is to give you more information about the graphs.
    With them, you will have a better understanding of the data in the graphs.
    In another moment, you will receive the outputs of this code, and you will use it to write a conclusion about the graphs you generated.
    - NEVER save a csv containing whole colums. The goal of this csv is to give you more information about the graph, not to give you the whole data.
    - Do not print the tables, save them as csv files in the directory data/graph_to_text{i}.csv, where i is the number of the graph. Use pd.to_csv().
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

    error_list = True
    while error_list:
        logging.info("Sending request to OpenAI API...")
        response = call_gpt(
            messages=messages,
            model=Config.get('MODEL').get('ANALYSIS'),
            presence_penalty=1.2
        )
        logging.info("Respose received!")

        # Extract the response from the API
        json_string = response.choices[0].message.content.strip()

        try:
            # Convert the json string to a dictionary
            json_dict = json.loads(json_string)
        except Exception as e:
            # If there is an error, send the error message back to ChatGPT
            logging.info("Error: ", e, " in the following json string:", json_string)
            messages.append({"role": "assistant", "content": json_string})
            messages.append({"role": "user", "content": str(e) + "Fix the error and return ONLY the json string."})

            logging.info("Sending request to OpenAI API...")
            logging.info(messages)
            response = call_gpt(
                messages=messages,
                model=Config.get('MODEL').get('ANALYSIS'),
                temperature=0.1
            )

            # Extract the response from the API
            json_string = response.choices[0].message.content.strip()
            json_dict = json.loads(json_string)

        error_list = []
        for i in range(1,4):
            # get the graph code
            graph_code = json_dict[f"graph{i}"]["graph_code"]
            # run the code
            logging.info("Running the graph code:")
            logging.info(graph_code)

            # clear the plot
            plt.clf()
            # run the code
            try:
                exec(graph_code)
            except Exception as e:
                # turn e into a string that is json serializable
                e = str(e)
                logging.info(f"Error: {e}")
                traceback.print_exc()
                error_list.append([f"graph{i}", f"Error: {e}"])

            # if there is an error, send the error message back to ChatGPT
            if error_list:
                messages.append({"role": "assistant", "content": json_string})

                error_list_content = "here is a list of errors:\n"
                for error in error_list:
                    error_list_content += f"{error[0]}: {error[1]}\n"
                error_list_content += "Please fix the errors and send the whole json(with the 3 graphs) again."

                messages.append({"role": "user", "content": str(error_list)})
                logging.info("There was an error, sending the error message back to ChatGPT...")
                logging.info(messages[-1])

    # for now we will just return the json string
    return json_dict


def get_conclusion(analysis_report, df):
    # Initialize an empty dictionary to store the conclusions
    conclusions = {}

    # Initialize an empty string to store the user prompt
    user_prompt = ""
    messages = analysis_report["messages"]

    messages[0] = {"role": "system", "content": "You, as AnalysisGPT, have to describe a CSV file like a data analyst, but you will only receive a sample of 20 rows, even though you'll know the shape of the full dataset. You can't mention that you're analyzing a sample. You must summarize the data in the CSV file in natural language and talk about what you think about the dataset, what it is for, and so on. You must use HTML (starting in <h3>) to format your response, and your initial description will be used to generate a JSON with graphs and tables."}
    messages.append({"role": "assistant", "content": analysis_report["description"]})
    messages.append({"role": "user", "content": "Now genarate a JSON containing 3 graphs, each one with a graph_code, that will generate a graph image based on the df. Also a graph_to_text_code that will generate csvs with the same information as the graphs. The csvs will be used by you to generate conclusions regarding the whole dataset."})
    messages.append({"role": "assistant", "content": str(analysis_report["analysis"])})
    # Iterate through the graphs in the analysis_report
    for graph_key, graph_value in analysis_report["analysis"].items():
        # Get the graph_to_text_code for the current graph
        graph_to_text_code = graph_value["graph_to_text_code"]

        # Run the graph_to_text_code
        try:
            logging.info(f"Running {graph_key} graph_to_text_code:")
            logging.info(graph_to_text_code)
            exec(graph_to_text_code)
        except Exception as e:
            logging.info(f"Error in {graph_key} graph_to_text_code: {e}")
            traceback.print_exc()
            conclusions[graph_key] = f"Error in {graph_key} graph_to_text_code: {e}"
            continue

        # Read the generated CSV file
        csv_path = f"data/graph_to_text{graph_key[-1]}.csv"
        try:
            graph_to_text_df = pd.read_csv(csv_path)
        except Exception as e:
            logging.info(f"Error reading {csv_path}: {e}")
            traceback.print_exc()
            conclusions[graph_key] = f"Error reading {csv_path}: {e}"
            continue

        # Append the CSV content to the user prompt
        user_prompt += f"\n\n{graph_key}:\n{graph_to_text_df.to_csv(index=False)}"

    # Create a prompt for the GPT system
    prompt = f"""You're still AnalysisGPT. Your mission now is to analyze the data generated by graph_to_text_code provide a conclusion based on the data.

{user_prompt}

Please provide a conclusion in natural language, discussing the insights and patterns you observe in the data. DO NOT CREATE A CONCLUSION FOR EACH CELL, but rather a conclusion for the whole dataset.
If you want to format the text for better readability, you can use html tags."""

    messages.append({"role": "user", "content": prompt})

    # Send the request to the OpenAI API
    logging.info("Sending request to OpenAI API...")
    tokens = num_tokens_from_string(prompt, "gpt-3.5-turbo")
    logging.info(f"Number of tokens in prompt: {tokens}")
    try:
        response = call_gpt(
            model=Config.get('MODEL').get('CONCLUSION'),
            messages=messages
        )
    except Exception as e:
        logging.info(f"Error while sending request to OpenAI API: {e}")
        traceback.print_exc()
        return {"error": f"Error while sending request to OpenAI API: {e}"}
    logging.info("Respose received!")
    # Extract the response from the API
    conclusion = response.choices[0].message.content.strip()

    # Store the conclusion in the conclusions dictionary
    conclusions["conclusion"] = conclusion

    return conclusions


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

def call_gpt(messages: list, model: str, temperature: float = 1, presence_penalty: float = 0):
    """This function will call the OpenAI API and return the response"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        presence_penalty=presence_penalty
    )
    return response