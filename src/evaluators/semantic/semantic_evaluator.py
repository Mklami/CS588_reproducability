import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        Also handles potential whitespace issues after tag removal.
        """
        if pd.isna(text):
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove <summary> tags and any other XML tags
        text = re.sub(r'<summary>|</summary>|<[^>]+>', '', text)
        
        # Clean up excessive whitespace that might be left after tag removal
        text = ' '.join(text.split())
        
        return text.strip()
        
    def preprocess(self, text, verbose=False):
        """
        Preprocess the comment text with improved XML and whitespace handling.
        """
        if pd.isna(text):
            return ""
        
        if verbose:
            print("\nPreprocessing steps:")
            print("Original text:", text)
            
        # First remove XML tags
        text = self.clean_xml_tags(text)
        if verbose:
            print("After XML removal:", text)
            
        # Early return if text is empty after tag removal
        if not text.strip():
            return ""
            
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
            
        # Rejoin tokens, ensuring no empty strings
        result = ' '.join(token for token in tokens if token)
        if verbose:
            print("Final result:", result)
            
        return result

class SemanticSimilarityEvaluator:
    def __init__(self, model_name='microsoft/codebert-base'):
        """Initialize with CodeBERT model for code-specific embeddings."""
        print(f"Initializing with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.preprocessor = CommentPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def get_embedding(self, text, verbose=False):
        """Generate BERT embedding for a given text."""
        # Preprocess text
        text = self.preprocessor.preprocess(text, verbose=verbose)
        
        # Handle empty text
        if not text.strip():
            return torch.zeros(768, device=self.device)
            
        # Tokenize and encode
        inputs = self.tokenizer(text, 
                              return_tensors='pt', 
                              padding=True, 
                              truncation=True, 
                              max_length=512).to(self.device)
        
        # Get BERT embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[0][0]
        
    def compute_similarity(self, text1, text2, verbose=False):
        """Compute cosine similarity between two texts."""
        if verbose:
            print("\nProcessing first text:")
        emb1 = self.get_embedding(text1, verbose=verbose)
        
        if verbose:
            print("\nProcessing second text:")
        emb2 = self.get_embedding(text2, verbose=verbose)
        
        similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), 
                                                         emb2.unsqueeze(0))
        
        return similarity.item()

def evaluate_comments(input_file, output_file, show_preprocessing_examples=3):
    """
    Evaluate semantic similarity between comments.
    """
    try:
        # Read the matched comments
        print("\nReading matched comments file...")
        df = pd.read_csv(input_file)
        
        # Initialize evaluator
        evaluator = SemanticSimilarityEvaluator()
        
        # Show preprocessing examples
        print("\nPreprocessing Examples:")
        for i in range(min(show_preprocessing_examples, len(df))):
            print(f"\n=== Example {i+1} ===")
            print("\nLLM Comment:")
            evaluator.compute_similarity(df.iloc[i]['LLM_Comment'], 
                                      df.iloc[i]['LLM_Comment'], 
                                      verbose=True)
            
            print("\nNon-LLM Comment:")
            evaluator.compute_similarity(df.iloc[i]['NonLLM_Comment'], 
                                      df.iloc[i]['NonLLM_Comment'], 
                                      verbose=True)
            
            print("\nOriginal Comment:")
            evaluator.compute_similarity(df.iloc[i]['Original_Comment'], 
                                      df.iloc[i]['Original_Comment'], 
                                      verbose=True)
        
        # Calculate similarities
        print("\nCalculating semantic similarities...")
        similarities = {
            'LLM_vs_NonLLM': [],
            'LLM_vs_Original': [],
            'NonLLM_vs_Original': []
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Calculate similarities between all pairs
            llm_nonllm = evaluator.compute_similarity(row['LLM_Comment'], row['NonLLM_Comment'])
            llm_original = evaluator.compute_similarity(row['LLM_Comment'], row['Original_Comment'])
            nonllm_original = evaluator.compute_similarity(row['NonLLM_Comment'], row['Original_Comment'])
            
            similarities['LLM_vs_NonLLM'].append(llm_nonllm)
            similarities['LLM_vs_Original'].append(llm_original)
            similarities['NonLLM_vs_Original'].append(nonllm_original)
        
        # Add similarities to dataframe
        df['Similarity_LLM_NonLLM'] = similarities['LLM_vs_NonLLM']
        df['Similarity_LLM_Original'] = similarities['LLM_vs_Original']
        df['Similarity_NonLLM_Original'] = similarities['NonLLM_vs_Original']
        
        # Calculate statistics
        stats = {}
        for comparison, scores in similarities.items():
            stats[comparison] = {
                'Mean': np.mean(scores),
                'Median': np.median(scores),
                'Std Dev': np.std(scores),
                'Min': np.min(scores),
                'Max': np.max(scores)
            }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.hist([similarities['LLM_vs_NonLLM'], 
                 similarities['LLM_vs_Original'],
                 similarities['NonLLM_vs_Original']], 
                label=['LLM vs NonLLM', 'LLM vs Original', 'NonLLM vs Original'],
                bins=20, alpha=0.6)
        plt.title('Distribution of Semantic Similarities')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('similarity_distributions_v3.png')
        plt.close()
        
        # Save results
        df.to_csv(output_file, index=False)
        
        # Print summary
        print("\nSemantic Similarity Statistics:")
        for comparison, metrics in stats.items():
            print(f"\n{comparison}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
        
        print(f"\nResults saved to: {output_file}")
        print("Similarity distribution plot saved as: similarity_distributions_v3.png")
        
        return df, stats
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None, None

if __name__ == "__main__":
    input_file = "matched_comments_with_original.csv"
    output_file = "semantic_similarity_results_v3.csv"
    
    results, statistics = evaluate_comments(input_file, output_file, show_preprocessing_examples=3)