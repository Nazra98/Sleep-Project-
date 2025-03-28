import re
import json
import pandas as pd
import openai
from openai import OpenAI

# OpenAI API key 
api_key =  "sk-proj-Z"
openai.api_key = api_key 

# Load and clean queries from file

#file_path = "Category 1 Prompt.txt"
file_path = "Category 1 No Prompt.txt"

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean up the text and extract queries
cleaned_text = re.sub(r'[\r\n\t]', ' ', raw_text)
parsed_data = json.loads(cleaned_text) if raw_text.strip().startswith("{") else {"note": raw_text}
extracted_queries = re.findall(r'\d+\.\s*(.*?)\s*(?=\d+\.|\Z)', parsed_data.get("note", ""), re.DOTALL)

def fetch_response(query):
    """Send a query to OpenAI and return the response."""
    client = OpenAI(
        api_key=api_key
     )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
              # Prompt {"role": "system", "content": "You are a sleep expert who answers questions using scientific accuracy and AASM guidelines."}, 
              
              {"role": "user", "content": [{"type": "text", "text": query}
                                           
             ]
          }
        ],

    )
    except Exception as e:
        return f"Error: {str(e)}"
    return response.choices[0].message.content

# Limit number of queries to avoid exceeding API limits
num_queries_to_process = min(10, len(extracted_queries)) 
responses = [fetch_response(query) for query in extracted_queries[:num_queries_to_process]]

# Store results
results = [{"query": q, "response": r} for q, r in zip(extracted_queries[:num_queries_to_process], responses)]

# Save results to CSV file 
df_results = pd.DataFrame(results)
df_results.to_csv("Category 1 No Prompt.csv", index=False, encoding="utf-8")
print("Batch processing complete! Results saved to 'Category 1 No Prompt.csv'")
