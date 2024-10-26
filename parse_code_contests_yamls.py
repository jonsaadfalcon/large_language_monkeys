import os
from datasets import Dataset
from tqdm import tqdm
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_test_matrix(content: str) -> List[List[bool]]:
    """
    Parse the unit_tests_passed_individual_scores matrix from the content.
    Returns a list of 1000 lists, where each inner list contains 20 boolean values.
    """
    lines = content.split('\n')
    raw_matrix = []
    current_row = None
    parsing_unit_tests = False
    
    # First, collect all rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'unit_tests_passed_individual_scores:' in line:
            parsing_unit_tests = True
            continue
            
        if not parsing_unit_tests:
            continue
            
        # Start of a new row
        if line.startswith('- - '):
            if current_row is not None:
                raw_matrix.append(current_row)
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
        elif line == '- null':
            if current_row is not None:
                raw_matrix.append(current_row)
            raw_matrix.append(None)
            current_row = None
            
    # Add the last row if exists
    if current_row is not None:
        raw_matrix.append(current_row)

    # Get the number of samples from the first valid row
    num_samples = 0
    for row in raw_matrix:
        if row is not None:
            num_samples = len(row)
            break

    if num_samples == 0:
        raise ValueError("No valid data found in the file")

    # Get the number of tests (non-None rows)
    num_tests = sum(1 for row in raw_matrix if row is not None)
    
    logging.info(f"Found {num_samples} samples and {num_tests} tests")
    
    # Initialize result matrix
    result_matrix = []
    for _ in range(num_samples):
        result_matrix.append([False] * num_tests)
    
    # Fill in the results
    test_idx = 0
    for row in raw_matrix:
        if row is not None:  # Skip None rows
            for sample_idx, result in enumerate(row):
                if sample_idx < num_samples:
                    result_matrix[sample_idx][test_idx] = result
            test_idx += 1
    
    return result_matrix

def process_file(file_path: str) -> List[List[bool]]:
    """Process a single file and return the test results matrix."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return parse_test_matrix(content)

def process_directory(directory_path: str) -> Dataset:
    """
    Process all files in directory and create a dataset with unit test results.
    Returns a dataset with a single column 'unit_tests_passed' containing the test matrices.
    """
    logging.info(f"Processing files in directory: {directory_path}")
    
    # List all files in directory
    try:
        all_files = os.listdir(directory_path)
        logging.info(f"Found {len(all_files)} files in directory")
    except Exception as e:
        logging.error(f"Error reading directory {directory_path}: {str(e)}")
        raise e

    # Process each file
    all_test_results = []
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
            
        try:
            test_results = process_file(file_path)
            all_test_results.extend(test_results)
            
            logging.info(f"\nProcessed file: {filename}")
            logging.info(f"Number of samples: {len(test_results)}")
            logging.info(f"Tests per sample: {len(test_results[0])}")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            raise e

    if not all_test_results:
        raise ValueError("No data was processed successfully!")
    
    # Create dataset with single column
    dataset = Dataset.from_dict({
        'unit_tests_passed': all_test_results
    })
    
    logging.info(f"Created dataset with {len(dataset)} samples")
    return dataset

def main(input_directory: str, output_filepath: str):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    dataset = process_directory(input_directory)
    
    logging.info(f"Saving dataset to {output_filepath}")
    dataset.save_to_disk(output_filepath)
    
    logging.info(f"Dataset saved successfully. Total samples: {len(dataset)}")
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