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

    breakpoint()
    
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

    # The raw matrix is 20 x 1000 (20 tests, 1000 samples)
    # We need to transpose it to get 1000 x 20 (1000 samples, 20 tests each)
    
    num_tests = 20  # We expect 20 non-None test rows
    num_samples = 1000  # We expect 1000 samples
    
    # Initialize the result matrix
    result_matrix = [[False] * num_tests for _ in range(num_samples)]
    
    # Count valid test rows and their indices
    valid_test_indices = []
    for i, row in enumerate(raw_matrix):
        if row is not None:
            valid_test_indices.append(i)
    
    if len(valid_test_indices) != num_tests:
        logging.warning(f"Expected {num_tests} valid test rows, but found {len(valid_test_indices)}")
    
    # Fill in the results by transposing the valid rows
    for test_idx, matrix_idx in enumerate(valid_test_indices[:num_tests]):
        row = raw_matrix[matrix_idx]
        if row is None:
            continue
            
        # Handle each sample
        for sample_idx in range(min(len(row), num_samples)):
            result_matrix[sample_idx][test_idx] = row[sample_idx]
    
    logging.info(f"Processed matrix with {len(result_matrix)} samples, {len(result_matrix[0])} tests per sample")
    return result_matrix

def process_file(file_path: str) -> List[List[bool]]:
    """Process a single file and return the test results matrix."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return parse_test_matrix(content)

def process_directory(directory_path: str) -> Dataset:
    """
    Process all files in directory and create a dataset.
    Returns a dataset with a single row containing a list of 1000 lists of 20 boolean values each.
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
    all_results = None
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
            
        try:
            test_results = process_file(file_path)
            all_results = test_results  # We only expect one file
            
            logging.info(f"\nProcessed file: {filename}")
            logging.info(f"Number of samples: {len(test_results)}")
            logging.info(f"Tests per sample: {len(test_results[0])}")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            raise e

    if all_results is None:
        raise ValueError("No data was processed successfully!")
    
    # Create dataset with a single row containing all results
    dataset = Dataset.from_dict({
        'unit_tests_passed': [all_results]  # Wrap in list to create single row
    })
    
    logging.info(f"Created dataset with {len(dataset)} rows")
    return dataset

def main(input_directory: str, output_filepath: str):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    dataset = process_directory(input_directory)
    
    logging.info(f"Saving dataset to {output_filepath}")
    dataset.save_to_disk(output_filepath)
    
    logging.info(f"Dataset saved successfully. Total samples: {len(dataset)}")
    breakpoint()

    true_found = "true" in str(dataset["unit_tests_passed"]).lower()
    print("True Found:", true_found)

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