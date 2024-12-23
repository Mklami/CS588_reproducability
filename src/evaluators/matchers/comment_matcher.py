import pandas as pd
import hashlib
import difflib
import re

def normalize_method_aggressive(method_text):
    """
    More aggressive normalization of method text to catch more matches.
    """
    if pd.isna(method_text):
        return ""
    
    # Convert to string if not already
    method_text = str(method_text)
    
    # Convert to lowercase
    method_text = method_text.lower()
    
    # Remove all whitespace
    method_text = ''.join(method_text.split())
    
    # Remove common variations in syntax
    method_text = re.sub(r'[\s\{\}\(\);]', '', method_text)
    
    # Remove common modifiers that might vary
    method_text = re.sub(r'(public|private|protected|internal|static|virtual|override|abstract)', '', method_text)
    
    return method_text

def analyze_unmatched_methods(df1, df2, df3, normalized_col='Method_Normalized'):
    """
    Analyze methods that didn't match across the datasets.
    """
    all_methods_1 = set(df1[normalized_col])
    all_methods_2 = set(df2[normalized_col])
    all_methods_3 = set(df3[normalized_col])
    
    # Find methods unique to each dataset
    unique_to_1 = all_methods_1 - (all_methods_2 | all_methods_3)
    unique_to_2 = all_methods_2 - (all_methods_1 | all_methods_3)
    unique_to_3 = all_methods_3 - (all_methods_1 | all_methods_2)
    
    # Find methods present in pairs but not all three
    in_1_2_not_3 = (all_methods_1 & all_methods_2) - all_methods_3
    in_1_3_not_2 = (all_methods_1 & all_methods_3) - all_methods_2
    in_2_3_not_1 = (all_methods_2 & all_methods_3) - all_methods_1
    
    print("\nUnmatched Analysis:")
    print(f"Methods only in LLM file: {len(unique_to_1)}")
    print(f"Methods only in Non-LLM file: {len(unique_to_2)}")
    print(f"Methods only in Original file: {len(unique_to_3)}")
    print(f"Methods in LLM & Non-LLM but not Original: {len(in_1_2_not_3)}")
    print(f"Methods in LLM & Original but not Non-LLM: {len(in_1_3_not_2)}")
    print(f"Methods in Non-LLM & Original but not LLM: {len(in_2_3_not_1)}")
    
    return {
        'unique_to_llm': unique_to_1,
        'unique_to_nonllm': unique_to_2,
        'unique_to_original': unique_to_3,
        'in_llm_nonllm_not_original': in_1_2_not_3,
        'in_llm_original_not_nonllm': in_1_3_not_2,
        'in_nonllm_original_not_llm': in_2_3_not_1
    }

def find_similar_methods(method, method_set, threshold=0.85):
    """
    Find similar methods using difflib's SequenceMatcher.
    """
    similar_methods = []
    for other_method in method_set:
        similarity = difflib.SequenceMatcher(None, method, other_method).ratio()
        if similarity >= threshold:
            similar_methods.append((other_method, similarity))
    return similar_methods

def compare_and_merge_all_comments(llm_file, nonllm_file, original_file, output_file, similarity_threshold=0.85):
    """
    Compare methods from all three files with enhanced matching and analysis.
    """
    try:
        # Read the CSV files
        print("\nReading input files...")
        llm_df = pd.read_csv(llm_file)
        nonllm_df = pd.read_csv(nonllm_file)
        original_df = pd.read_csv(original_file)
        
        print(f"\nInitial statistics:")
        print(f"LLM file rows: {len(llm_df)}")
        print(f"Non-LLM file rows: {len(nonllm_df)}")
        print(f"Original file rows: {len(original_df)}")
        
        # Apply both normal and aggressive normalization
        print("\nApplying normalizations...")
        for df in [llm_df, nonllm_df, original_df]:
            df['Method_Normalized'] = df['Method'].apply(normalize_method_aggressive)
        
        # Remove duplicates
        llm_df_unique = llm_df.drop_duplicates(subset='Method_Normalized', keep='first')
        nonllm_df_unique = nonllm_df.drop_duplicates(subset='Method_Normalized', keep='first')
        original_df_unique = original_df.drop_duplicates(subset='Method_Normalized', keep='first')
        
        # Find the comment column in non-LLM file
        nonllm_comment_col = 'GeneratedComment' if 'GeneratedComment' in nonllm_df.columns else 'Generated Summary'
        
        # Merge all three datasets
        print("\nMerging datasets...")
        merged_df = pd.merge(
            llm_df_unique[['Method', 'Method_Normalized', 'Generated Summary']],
            nonllm_df_unique[['Method_Normalized', nonllm_comment_col]],
            on='Method_Normalized',
            how='inner'
        )
        
        final_df = pd.merge(
            merged_df,
            original_df_unique[['Method_Normalized', 'Summary']],
            on='Method_Normalized',
            how='inner'
        )
        
        # Analyze unmatched methods
        unmatched = analyze_unmatched_methods(llm_df_unique, nonllm_df_unique, original_df_unique)
        
        # Create sample of unmatched methods with near matches
        print("\nAnalyzing near matches (samples)...")
        for category, methods in unmatched.items():
            if methods:
                sample_method = next(iter(methods))
                if category.startswith('unique_to_'):
                    other_methods = set.union(*(set(df['Method_Normalized']) for df in [llm_df_unique, nonllm_df_unique, original_df_unique])) - {sample_method}
                    similar = find_similar_methods(sample_method, other_methods)
                    if similar:
                        print(f"\nSample near match for {category}:")
                        print(f"Method: {sample_method[:100]}...")
                        print("Similar to:")
                        for similar_method, sim_score in similar[:2]:
                            print(f"- {similar_method[:100]}... (similarity: {sim_score:.2f})")
        
        # Create the final dataframe
        result_df = pd.DataFrame({
            'Method': final_df['Method'],
            'LLM_Comment': final_df['Generated Summary'],
            'NonLLM_Comment': final_df[nonllm_comment_col],
            'Original_Comment': final_df['Summary']
        })
        
        # Save results
        result_df.to_csv(output_file, index=False)
        
        print(f"\nFinal Results:")
        print(f"Successfully matched methods across all three sources: {len(result_df)}")
        print(f"Results saved to: {output_file}")
        
        return result_df, unmatched
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None, None

if __name__ == "__main__":
    llm_file = "/Users/mayasalami/Desktop/dataset_construct/500_methods_with_generated_comments.csv"      
    nonllm_file = "/Users/mayasalami/commentGenerate/500_comment_generated_nonllm.csv"
    original_file = "/Users/mayasalami/Desktop/dataset_construct/500_methods_and_summaries.csv"
    output_file = "matched_comments_with_original.csv"
    
    result, unmatched = compare_and_merge_all_comments(llm_file, nonllm_file, original_file, output_file)