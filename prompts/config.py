import os
from dotenv import load_dotenv
load_dotenv()

import mlflow
mlflow.set_tracking_uri("http://20.75.92.162:5000/")


# Use double curly braces for variables in the template
company_stock_code_prompt = """what is the stock code for given company, 
                                              Provide only stock code
                                              Strictly Donot provide any other details, If company name is not valid, ask valid.  
                                              company name: {company_name}
                                            """ 
# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="akshay_stock_code_generator",
    template=company_stock_code_prompt,
    # Optional: Provide a commit message to describe the changes
    commit_message="Initial commit",
    # Optional: Set tags applies to the prompt (across versions)
    tags={
        "author": "Akshay T",
        "task": "stock_code_generation",
        "language": "en",
        'llm': 'gpt-4o-mini'
    },
)

# The prompt object contains information about the registered prompt
print(f"Created prompt '{prompt.name}' (version {prompt.version})")



sentiment_analyser_prompt ="""
                                        You will be given a list of news summaries.

                                        For each summary, perform the following tasks:
                                        1. Classify Sentiment: Choose one of [Positive, Negative, Neutral].
                                        2. Extract Named Entities: Identify and list unique mentions of:
                                        - People (e.g., executives, politicians, public figures)
                                        - Places (cities, countries, regions)
                                        - Companies/Organizations (excluding the source itself)
                                        3. Return Structured JSON using the schema below.

                                        News summaries:
                                        {news_op}

                                        Formatting requirements:
                                        - Always output a JSON array of objects, one per summary.
                                        - If a category has no entities, return an empty array [].
                                        - Do not include explanations or text outside of the JSON array.

                                        Output schema:
                                        [
                                        {{
                                            "summary": string,          // original news summary
                                            "sentiment": string,        // Positive, Negative, or Neutral
                                            "entities": {{
                                                "people": [string],     // list of people mentioned
                                                "places": [string],     // list of places mentioned
                                                "companies": [string]   // list of companies/organizations mentioned
                                            }}
                                        }}
                                        ]
                                        """

# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="akshay_sentiment_analyser",
    template=sentiment_analyser_prompt,
    # Optional: Provide a commit message to describe the changes
    commit_message="Initial commit",
    # Optional: Set tags applies to the prompt (across versions)
    tags={
        "author": "Akshay T",
        "task": "company news sentiment analysis",
        "language": "en",
        'llm': 'gpt-4o-mini'
    },
)

# The prompt object contains information about the registered prompt
print(f"Created prompt '{prompt.name}' (version {prompt.version})")
