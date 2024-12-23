import csv
import time
import requests
from typing import List
from dataclasses import dataclass
import sys
import json

@dataclass
class MethodEntry:
    method: str
    generated_summary: str = ""

class CommentGenerator:
    def __init__(self, model_name: str = "codellama"):
        """Initialize with Ollama model name."""
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api/generate"
        self.retry_delay = 20
        self.max_retries = 3
        
        # Check if Ollama is running and accessible
        self.check_ollama_status()
        
    def check_ollama_status(self):
        """Check if Ollama is running and the model is available."""
        try:
            # Try to connect to Ollama
            response = requests.get("http://localhost:11434/")
            if response.status_code != 200:
                print("Warning: Ollama API seems to be running but returned unexpected status code")
                
            # Check if model exists
            model_response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model_name, "prompt": "test", "stream": False}
            )
            
            if model_response.status_code == 404:
                print(f"\nError: Model '{self.model_name}' not found!")
                print(f"Please run: ollama pull {self.model_name}")
                sys.exit(1)
                
        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to Ollama!")
            print("Please ensure that:")
            print("1. Ollama is installed (https://ollama.ai/)")
            print("2. Ollama is running (run 'ollama serve' in terminal)")
            print(f"3. The model is pulled (run 'ollama pull {self.model_name}')")
            sys.exit(1)

    def generate_comment(self, method: str) -> str:
        """Generate a comment for a given method using Code Llama."""
        prompt = f"""As a technical documentation expert, create a concise XML documentation comment for the following C# method.
        The comment should follow C# XML documentation format with <summary> tags.
        Only return the comment itself, nothing else.
        
        Method:
        {method}
        """

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    comment = response.json()['response'].strip()
                    
                    if not comment.startswith("<summary>"):
                        comment = f"<summary>\n{comment}\n</summary>"
                    elif not comment.endswith("</summary>"):
                        comment = f"{comment}\n</summary>"
                    
                    return comment
                else:
                    error_msg = f"Error from Ollama API: {response.status_code}"
                    if hasattr(response, 'text'):
                        error_msg += f" - {response.text}"
                    print(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue

            except requests.exceptions.Timeout:
                print("Request timed out. Retrying...")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
            except Exception as e:
                print(f"Error generating comment: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return ""

        return ""

def count_rows(file_path: str) -> int:
    """Count the number of valid rows in the CSV file."""
    valid_rows = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Method') and row.get('Method').strip():  # Check for non-empty Method field
                valid_rows += 1
    return valid_rows

def save_progress(methods: List[MethodEntry], output_file: str):
    """Save the processed methods to a CSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['Generated Summary', 'Method'])
            
            for method in methods:
                if method.generated_summary:  # Only save if we have a generated summary
                    writer.writerow([method.generated_summary, method.method])
                    
        print(f"Successfully saved {len(methods)} methods to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {str(e)}")
        # Create a backup file
        backup_file = output_file + '.error_backup'
        with open(backup_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['Generated Summary', 'Method'])
            for method in methods:
                if method.generated_summary:
                    writer.writerow([method.generated_summary, method.method])
        print(f"Backup saved to {backup_file}")

def process_methods_file(input_file: str, output_file: str, model_name: str = "codellama"):
    """Process a file containing C# methods and generate comments for each."""
    generator = CommentGenerator(model_name)
    processed_methods = []
    
    # Get total number of methods first
    total_methods = count_rows(input_file)
    print(f"Processing {total_methods} methods...")
    
    # Read and process the input CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader, 1):
            method = row['Method']
            print(f"Processing method {i} of {total_methods}")
            
            try:
                # Generate comment
                generated_summary = generator.generate_comment(method)
                if generated_summary:  # Only add if we got a valid comment
                    processed_methods.append(MethodEntry(
                        method=method,
                        generated_summary=generated_summary
                    ))
                
                # Save progress periodically
                if i % 10 == 0:
                    save_progress(processed_methods, output_file)
                    print(f"Progress saved. Completed {i}/{total_methods} methods.")
                    
            except Exception as e:
                print(f"Error processing method {i}: {str(e)}")
                # Save progress on error
                if processed_methods:
                    save_progress(processed_methods, f"{output_file}.backup")
                continue

    # Save final results
    save_progress(processed_methods, output_file)
    print("Processing complete!")

#main
if __name__ == "__main__":
    input_file = "500_methods.csv"
    output_file = "500_methods_with_generated_comments.csv"
    model_name = "codellama"  # or "codellama:13b" or "codellama-instruct"
    
    print(f"\nInitializing with model: {model_name}")
    print("Checking Ollama status...")
    
    try:
        process_methods_file(input_file, output_file, model_name)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
