import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import math

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ReadabilityEvaluator:
    def __init__(self):
        self.complex_word_threshold = 3  # Words with more than 3 syllables are complex
    
    def clean_text(self, text):
        """Remove XML tags and clean the text."""
        if pd.isna(text):
            return ""
        
        # Remove XML tags
        text = re.sub(r'<summary>|</summary>|<[^>]+>', '', str(text))
        
        # Clean up whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def count_syllables(self, word):
        """Count the number of syllables in a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        on_vowel = False
        last_char = None
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
            last_char = char
            
        # Handle special cases
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count = 1
            
        return count
    
    def is_complex_word(self, word):
        """Determine if a word is complex (more than 3 syllables)."""
        return self.count_syllables(word) > self.complex_word_threshold
    
    def count_sentences(self, text):
        """Count the number of sentences in the text."""
        text = self.clean_text(text)
        if not text:
            return 0
        return len(sent_tokenize(text))
    
    def count_words(self, text):
        """Count the number of words in the text."""
        text = self.clean_text(text)
        if not text:
            return 0
        words = word_tokenize(text)
        return len([word for word in words if word.isalnum()])
    
    def calculate_flesch_kincaid(self, text):
        """Calculate Flesch-Kincaid Grade Level."""
        text = self.clean_text(text)
        if not text:
            return 0
        
        sentences = sent_tokenize(text)
        if not sentences:
            return 0
            
        words = word_tokenize(text)
        word_count = len([word for word in words if word.isalnum()])
        if word_count == 0:
            return 0
            
        syllable_count = sum(self.count_syllables(word) for word in words if word.isalnum())
        
        # Flesch-Kincaid Grade Level = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        return 0.39 * (word_count / len(sentences)) + 11.8 * (syllable_count / word_count) - 15.59
    
    def calculate_smog(self, text):
        """Calculate SMOG Index."""
        text = self.clean_text(text)
        if not text:
            return 0
            
        sentences = sent_tokenize(text)
        if len(sentences) < 3:  # SMOG requires at least 3 sentences
            return 0
            
        words = word_tokenize(text)
        complex_words = len([word for word in words if word.isalnum() and self.is_complex_word(word)])
        
        # SMOG = 1.043 * sqrt(complex_words * (30 / sentences)) + 3.1291
        return 1.043 * math.sqrt(complex_words * (30 / len(sentences))) + 3.1291

def evaluate_readability(input_file, output_file):
    """
    Evaluate readability of comments using multiple metrics.
    """
    try:
        print("\nReading matched comments file...")
        df = pd.read_csv(input_file)
        
        evaluator = ReadabilityEvaluator()
        
        print("\nCalculating readability scores...")
        
        # Initialize result containers
        results = {
            'LLM': {'flesch_kincaid': [], 'smog': []},
            'NonLLM': {'flesch_kincaid': [], 'smog': []},
            'Original': {'flesch_kincaid': [], 'smog': []}
        }
        
        # Calculate scores for each comment
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Calculate scores for each type of comment
            for comment_type in ['LLM_Comment', 'NonLLM_Comment', 'Original_Comment']:
                system = comment_type.split('_')[0]
                text = row[comment_type]
                
                flesch_kincaid = evaluator.calculate_flesch_kincaid(text)
                smog = evaluator.calculate_smog(text)
                
                results[system]['flesch_kincaid'].append(flesch_kincaid)
                results[system]['smog'].append(smog)
        
        # Add scores to dataframe
        for system in ['LLM', 'NonLLM', 'Original']:
            df[f'{system}_Flesch_Kincaid'] = results[system]['flesch_kincaid']
            df[f'{system}_SMOG'] = results[system]['smog']
        
        # Calculate statistics
        stats = {}
        for system in ['LLM', 'NonLLM', 'Original']:
            stats[system] = {
                'Flesch_Kincaid': {
                    'Mean': np.mean(results[system]['flesch_kincaid']),
                    'Median': np.median(results[system]['flesch_kincaid']),
                    'Std Dev': np.std(results[system]['flesch_kincaid'])
                },
                'SMOG': {
                    'Mean': np.mean(results[system]['smog']),
                    'Median': np.median(results[system]['smog']),
                    'Std Dev': np.std(results[system]['smog'])
                }
            }
        
        # Create visualizations
        metrics = ['Flesch_Kincaid', 'SMOG']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            data = [results[system][metric.lower()] for system in ['LLM', 'NonLLM', 'Original']]
            
            plt.boxplot(data, labels=['LLM', 'NonLLM', 'Original'])
            plt.title(f'{metric} Grade Level Distribution')
            plt.ylabel('Grade Level')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f'{metric.lower()}_distribution.png')
            plt.close()
        
        # Save results
        df.to_csv(output_file, index=False)
        
        # Print summary statistics
        print("\nReadability Score Statistics:")
        for system in ['LLM', 'NonLLM', 'Original']:
            print(f"\n{system}:")
            for metric in ['Flesch_Kincaid', 'SMOG']:
                print(f"\n{metric}:")
                for stat_name, value in stats[system][metric].items():
                    print(f"{stat_name}: {value:.2f}")
        
        # Print example comparisons
        print("\nExample Comparisons (first 3 comments):")
        for idx in range(min(3, len(df))):
            print(f"\nExample {idx + 1}:")
            for comment_type in ['LLM_Comment', 'NonLLM_Comment', 'Original_Comment']:
                system = comment_type.split('_')[0]
                text = df.iloc[idx][comment_type]
                fk_score = df.iloc[idx][f'{system}_Flesch_Kincaid']
                smog_score = df.iloc[idx][f'{system}_SMOG']
                
                print(f"\n{comment_type}:")
                print(f"Text: {text}")
                print(f"Flesch-Kincaid Grade Level: {fk_score:.2f}")
                print(f"SMOG Index: {smog_score:.2f}")
        
        print(f"\nResults saved to: {output_file}")
        print("Visualizations saved as flesch_kincaid_distribution.png and smog_distribution.png")
        
        return df, stats
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None, None

if __name__ == "__main__":
    input_file = "matched_comments_with_original.csv"
    output_file = "readability_evaluation_results.csv"
    
    results, statistics = evaluate_readability(input_file, output_file)