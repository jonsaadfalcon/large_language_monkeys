import os
from datasets import Dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_test_scores(tests_matrix: List[List[Optional[bool]]]) -> Dict[str, float]:
    """
    Calculate scores for the test matrix.
    Returns dict with overall score and per-sample scores.
    """
    if not tests_matrix:
        return {'overall_score': 0.0, 'sample_scores': []}
        
    sample_scores = []
    for i in range(len(tests_matrix[0]) if tests_matrix[0] else 0):
        # Get the i-th element from each valid row
        valid_values = []
        for row in tests_matrix:
            if row is not None and i < len(row):
                if row[i] is not None:
                    valid_values.append(row[i])
        
        if valid_values:
            score = sum(1 for v in valid_values if v) / len(valid_values)
            sample_scores.append(score)
        else:
            sample_scores.append(0.0)
    
    overall_score = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    
    return {
        'overall_score': overall_score,
        'sample_scores': sample_scores
    }

def extract_data(content: str) -> Dict[str, Any]:
    """Extract data from the text content."""
    # Split content into lines and process
    lines = content.strip().split('\n')
    
    # Process each row
    all_rows = []
    current_row = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line == "- null":
            if current_row:
                all_rows.append(current_row)
            all_rows.append(None)  # Add null row
            current_row = []
        elif line.startswith('  - '):
            # Values in a row
            value = line[4:].strip()
            if value == "null":
                current_row.append(None)
            else:
                current_row.append(value == 'true')
        elif line.startswith('- -'):
            # New row
            if current_row:
                all_rows.append(current_row)
            current_row = []
    
    # Add the last row if it exists
    if current_row:
        all_rows.append(current_row)
    
    # Calculate scores
    scores = process_test_scores(all_rows)
    
    return {
        'unit_tests_passed': scores['overall_score'],
        'sample_scores': scores['sample_scores'],
        'tests_matrix': all_rows
    }

def process_directory(directory_path: str) -> List[Dict[str, Any]]:
    data = []
    
    logging.info(f"Processing files in directory: {directory_path}")
    
    # List all files in directory
    try:
        all_files = os.listdir(directory_path)
        logging.info(f"Found {len(all_files)} files in directory")
    except Exception as e:
        logging.error(f"Error reading directory {directory_path}: {str(e)}")
        return data

    # Process each file
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            result = extract_data(content)
            result['filename'] = filename
            data.append(result)
            
            # Log detailed scores for this file
            logging.info(f"\nFile: {filename}")
            logging.info(f"Overall score: {result['unit_tests_passed']:.3f}")
            logging.info("Sample scores:")
            for i, score in enumerate(result['sample_scores']):
                logging.info(f"Sample {i}: {score:.3f}")
                
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
    
    if not data:
        logging.error("No valid files were processed!")
    else:
        logging.info(f"Successfully processed {len(data)} files")
        
    # Log average scores across all files
    if data:
        overall_scores = [d['unit_tests_passed'] for d in data]
        avg_score = sum(overall_scores) / len(overall_scores)
        logging.info(f"\nAverage score across all files: {avg_score:.3f}")
        
        # Calculate and log average score per sample position
        num_samples = len(data[0]['sample_scores']) if data else 0
        if num_samples:
            logging.info("\nAverage scores by sample position:")
            for i in range(num_samples):
                sample_scores = [d['sample_scores'][i] for d in data if i < len(d['sample_scores'])]
                if sample_scores:
                    avg = sum(sample_scores) / len(sample_scores)
                    logging.info(f"Sample {i}: {avg:.3f}")
    
    return data

def create_dataset(data: List[Dict[str, Any]]) -> Dataset:
    logging.info("Creating Hugging Face dataset")
    if not data:
        raise ValueError("No data to create dataset from!")
    return Dataset.from_list(data)

def main(input_directory: str, output_filepath: str):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    data = process_directory(input_directory)
    
    if not data:
        raise ValueError("No data was processed successfully!")
    
    dataset = create_dataset(data)
    
    logging.info(f"Saving dataset to {output_filepath}")
    dataset.save_to_disk(output_filepath)
    
    logging.info(f"Dataset saved successfully. Total samples processed: {len(dataset)}")

if __name__ == "__main__":
    save_dir = os.environ.get('SAVE_DIR')
    if not save_dir:
        raise ValueError("SAVE_DIR environment variable is not set")

    input_directory = os.path.join(save_dir, "eval_results", "cc_samples_Llama-3.1-8B-Instruct_1000_samples_50_unit_tests_MIDWAY_v4")
    output_filepath = os.path.join(save_dir, "good_turing", "cc_samples_Llama-3.1-8B-Instruct_1000_samples_50_unit_tests_MIDWAY_v4.hf")

    logging.info(f"SAVE_DIR is set to: {save_dir}")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output filepath: {output_filepath}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    logging.info(f"Ensured output directory exists: {os.path.dirname(output_filepath)}")

    main(input_directory, output_filepath)