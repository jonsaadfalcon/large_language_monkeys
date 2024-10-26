import os
from datasets import Dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_test_matrix(content: str) -> List[List[Optional[bool]]]:
    """Parse the unit_tests_passed_individual_scores matrix from the content."""
    lines = content.split('\n')
    matrix = []
    current_row = None
    parsing_unit_tests = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'unit_tests_passed_individual_scores:' in line:
            parsing_unit_tests = True
            continue
            
        if not parsing_unit_tests:
            continue
            
        # Start of a new row for a test case
        if line.startswith('- - '):
            if current_row is not None:
                matrix.append(current_row)
            current_row = []
            value = line[4:].strip()
            if value == 'true':
                current_row.append(True)
            elif value == 'false':
                current_row.append(False)
            elif value == 'null':
                current_row = None
        # Continue current row
        elif line.startswith('  - '):
            if current_row is not None:
                value = line[4:].strip()
                if value == 'true':
                    current_row.append(True)
                elif value == 'false':
                    current_row.append(False)
                elif value == 'null':
                    current_row.append(None)
        elif line == '- null':
            if current_row is not None:
                matrix.append(current_row)
            matrix.append(None)
            current_row = None
            
    # Add the last row if exists
    if current_row is not None:
        matrix.append(current_row)
        
    return matrix

def calculate_sample_scores(matrix: List[List[Optional[bool]]]) -> List[float]:
    """Calculate score for each of the 1000 samples across their unit tests."""
    scores = []
    # Each column represents one unit test for a single sample
    num_samples = 20  # Each sample has 20 unit tests
    
    # Group the tests by sample (every 20 rows is for one sample)
    for i in range(0, len(matrix), num_samples):
        sample_rows = matrix[i:i+num_samples]
        if any(row is None for row in sample_rows):
            scores.append(0.0)  # Skip samples with null rows
            continue
            
        # Count true values for this sample
        total_true = 0
        total_valid = 0
        for row in sample_rows:
            if row is not None:
                for value in row:
                    if value is not None:
                        if value:
                            total_true += 1
                        total_valid += 1
        
        score = total_true / total_valid if total_valid > 0 else 0.0
        scores.append(score)
    
    return scores

def extract_data(content: str) -> Dict[str, Any]:
    """Extract data from text content."""
    matrix = parse_test_matrix(content)
    
    sample_scores = calculate_sample_scores(matrix)
    overall_score = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    
    return {
        'unit_tests_passed': overall_score,
        'sample_scores': sample_scores,
        'tests_matrix': matrix
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
            logging.info(f"Number of samples processed: {len(result['sample_scores'])}")
            logging.info("First 5 sample scores:")
            for i, score in enumerate(result['sample_scores'][:5]):
                logging.info(f"Sample {i}: {score:.3f}")
                
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            raise e  # Re-raise to see full traceback during development
    
    if not data:
        logging.error("No valid files were processed!")
    else:
        logging.info(f"Successfully processed {len(data)} files")
        
    # Log average scores across all files
    if data:
        overall_scores = [d['unit_tests_passed'] for d in data]
        avg_score = sum(overall_scores) / len(overall_scores)
        logging.info(f"\nAverage score across all files: {avg_score:.3f}")
    
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
    
    # Debugging information
    breakpoint()

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