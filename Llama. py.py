import re
import json
import pandas as pd
import requests

# Load and clean queries

file_path = "Category 1 Prompt.txt"
file_path = "Category 1 No Prompt.txt"

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean and extract numbered queries
cleaned_text = re.sub(r'[\r\n\t]', ' ', raw_text)
parsed_data = json.loads(cleaned_text) if raw_text.strip().startswith("{") else {"note": raw_text}
extracted_queries = re.findall(r'\d+\.\s*(.*?)\s*(?=\d+\.|\Z)', parsed_data.get("note", ""), re.DOTALL)

# Check if Ollama server is running
try:
    test = requests.get("http://localhost:11434/api/tags")
    test.raise_for_status()
except requests.exceptions.RequestException:
    print("Ollama is not running. Please start it using 'ollama run llama3.1'")
    exit()

# Query function for Ollama using /api/chat
def fetch_response_ollama(query, model="llama3.1"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a sleep expert who answers questions using scientific accuracy and AASM guidelines."},
                
            {"role": "user", "content": query}

        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Limit number of queries to process 
num_queries_to_process = min(10,len(extracted_queries))
responses = [fetch_response_ollama(query) for query in extracted_queries[:num_queries_to_process]]

# Store results
results = [{"query": q, "response": r} for q, r in zip(extracted_queries[:num_queries_to_process], responses)]

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("Category 1 Prompt.csv", index=False, encoding="utf-8")
print("Batch processing complete! Results saved to 'LLaMA3_Responses.csv'")
