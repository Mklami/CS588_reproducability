import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CommentPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_xml_tags(self, text):
        """
        Remove XML tags and standardize the text.
        """
        if pd.isna(text):
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove <summary> tags and any other XML tags
        text = re.sub(r'<summary>|</summary>|<[^>]+>', '', text)
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess(self, text, verbose=False):
        """
        Preprocess the comment text and return a set of tokens.
        """
        if pd.isna(text):
            return set()
            
        if verbose:
            print("\nOriginal text:", text)
            
        # Remove XML tags
        text = self.clean_xml_tags(text)
        if verbose:
            print("After XML removal:", text)
            
        # Early return if text is empty after tag removal
        if not text.strip():
            return set()
            
        # Convert to lowercase
        text = text.lower()
        if verbose:
            print("After lowercase:", text)
            
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        if verbose:
            print("After special char removal:", text)
            
        # Tokenize
        tokens = word_tokenize(text)
        if verbose:
            print("After tokenization:", tokens)
            
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.strip()]
        if verbose:
            print("After stopword removal and lemmatization:", tokens)
            
        return set(tokens)

def calculate_jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    Jaccard similarity = |intersection| / |union|
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def evaluate_precision(input_file, output_file, show_examples=3):
    """
    Evaluate precision using Jaccard similarity.
    """
    try:
        # Read the matched comments
        print("\nReading matched comments file...")
        df = pd.read_csv(input_file)
        
        # Initialize preprocessor
        preprocessor = CommentPreprocessor()
        
        # Show preprocessing examples
        print("\nPreprocessing Examples:")
        for i in range(min(show_examples, len(df))):
            print(f"\n=== Example {i+1} ===")
            print("\nLLM Comment:")
            preprocessor.preprocess(df.iloc[i]['LLM_Comment'], verbose=True)
            print("\nNon-LLM Comment:")
            preprocessor.preprocess(df.iloc[i]['NonLLM_Comment'], verbose=True)
            print("\nOriginal Comment:")
            preprocessor.preprocess(df.iloc[i]['Original_Comment'], verbose=True)
        
        print("\nCalculating Jaccard similarities...")
        
        # Calculate Jaccard similarities
        results = {
            'LLM_vs_Original': [],
            'NonLLM_vs_Original': []
        }
        
        # Process comments for each row
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Preprocess comments to token sets
            llm_tokens = preprocessor.preprocess(row['LLM_Comment'])
            nonllm_tokens = preprocessor.preprocess(row['NonLLM_Comment'])
            original_tokens = preprocessor.preprocess(row['Original_Comment'])
            
            # Calculate Jaccard similarities
            llm_jaccard = calculate_jaccard_similarity(llm_tokens, original_tokens)
            nonllm_jaccard = calculate_jaccard_similarity(nonllm_tokens, original_tokens)
            
            results['LLM_vs_Original'].append(llm_jaccard)
            results['NonLLM_vs_Original'].append(nonllm_jaccard)
        
        # Add results to dataframe
        df['Jaccard_LLM_Original'] = results['LLM_vs_Original']
        df['Jaccard_NonLLM_Original'] = results['NonLLM_vs_Original']
        
        # Calculate statistics
        stats = {}
        for comparison, scores in results.items():
            stats[comparison] = {
                'Mean': np.mean(scores),
                'Median': np.median(scores),
                'Std Dev': np.std(scores),
                'Min': np.min(scores),
                'Max': np.max(scores)
            }
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist([results['LLM_vs_Original'], results['NonLLM_vs_Original']], 
                label=['LLM vs Original', 'NonLLM vs Original'],
                bins=20, alpha=0.6)
        plt.title('Distribution of Jaccard Similarities')
        plt.xlabel('Jaccard Similarity Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('jaccard_similarities.png')
        plt.close()
        
        # Save results
        df.to_csv(output_file, index=False)
        
        # Print summary statistics
        print("\nJaccard Similarity Statistics:")
        for comparison, metrics in stats.items():
            print(f"\n{comparison}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
                
        # Print high/low similarity examples
        print("\nExample comparisons:")
        for comparison in ['LLM_vs_Original', 'NonLLM_vs_Original']:
            col_name = f'Jaccard_{comparison}'
            
            # Highest similarity example
            highest_idx = df[col_name].idxmax()
            highest = df.loc[highest_idx]
            
            # Lowest similarity example
            lowest_idx = df[col_name].idxmin()
            lowest = df.loc[lowest_idx]
            
            comment_type = 'LLM_Comment' if 'LLM' in comparison else 'NonLLM_Comment'
            
            print(f"\n{comparison} - Highest similarity (Score: {highest[col_name]:.3f}):")
            print(f"Generated: {highest[comment_type]}")
            print(f"Original:  {highest['Original_Comment']}")
            
            print(f"\n{comparison} - Lowest similarity (Score: {lowest[col_name]:.3f}):")
            print(f"Generated: {lowest[comment_type]}")
            print(f"Original:  {lowest['Original_Comment']}")
        
        return df, stats
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None, None

if __name__ == "__main__":
    input_file = "matched_comments_with_original.csv"
    output_file = "precision_evaluation_results.csv"
    
    results, statistics = evaluate_precision(input_file, output_file)