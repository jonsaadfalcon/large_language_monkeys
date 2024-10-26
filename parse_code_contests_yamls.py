import os
from datasets import Dataset
from tqdm import tqdm
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_test_matrix(content: str) -> List[List[bool]]:
    """
    Parse the unit_tests_passed_individual_scores matrix from the content.
    Returns a list of lists, where each inner list contains 20 boolean values.
    """
    # Split on the marker
    splits = content.split("unit_tests_passed_individual_scores:")
    if len(splits) != 2:
        raise ValueError(f"Expected 1 occurrence of 'unit_tests_passed_individual_scores:', found {len(splits)-1}")
    
    # Get the relevant section
    test_content = splits[1].strip()
    
    # Initialize containers
    matrix = []  # Will hold all samples
    current_sample = []  # Will hold current sample's test results
    true_count = 0
    false_count = 0
    
    # Process each line
    lines = test_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Start of a new sample
        if line.startswith('- - '):
            if current_sample:  # Store previous sample if it exists
                if len(current_sample) != 20:
                    logging.warning(f"Sample has {len(current_sample)} tests instead of 20")
                matrix.append(current_sample)
            current_sample = []  # Start new sample
            value = line[4:].strip()  # Remove '- - ' prefix
            
        # Continuation of current sample
        elif line.startswith('  - '):
            value = line[4:].strip()  # Remove '  - ' prefix
            
        # Skip null lines
        elif line == '- null':
            continue
            
        else:
            continue
            
        # Process the value
        if value == 'true':
            current_sample.append(True)
            true_count += 1
        elif value == 'false':
            current_sample.append(False)
            false_count += 1
            
    # Add the last sample if it exists
    if current_sample:
        if len(current_sample) != 20:
            logging.warning(f"Last sample has {len(current_sample)} tests instead of 20")
        matrix.append(current_sample)
    
    # Validation
    if not matrix:
        raise ValueError("No samples were parsed!")
        
    if true_count == 0:
        raise ValueError("No TRUE values found in parsing!")
        
    if false_count == 0:
        raise ValueError("No FALSE values found in parsing!")
    
    # Verify all samples have 20 tests
    for i, sample in enumerate(matrix):
        if len(sample) != 20:
            raise ValueError(f"Sample {i} has {len(sample)} tests instead of 20")
    
    # Log statistics
    logging.info(f"Parsing complete:")
    logging.info(f"  - Total samples parsed: {len(matrix)}")
    logging.info(f"  - Total TRUE values: {true_count}")
    logging.info(f"  - Total FALSE values: {false_count}")
    logging.info(f"  - Tests per sample: {len(matrix[0])}")
    
    return matrix

def process_file(file_path: str) -> List[List[bool]]:
    """Process a single file and return the test results matrix."""
    logging.info(f"Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    if not content:
        raise ValueError(f"File is empty: {file_path}")
        
    return parse_test_matrix(content)

def process_directory(directory_path: str) -> Dataset:
    """
    Process all files in directory and create a dataset.
    Returns a dataset with the test results.
    """
    logging.info(f"Processing files in directory: {directory_path}")
    
    # List all files in directory
    all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    logging.info(f"Found {len(all_files)} files in directory")
    
    for filename in all_files:
        logging.info(f"Found file: {filename}")

    # Process each file
    all_results = None
    processed_files = 0
    
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        logging.info(f"Attempting to process: {file_path}")
        
        try:
            test_results = process_file(file_path)
            all_results = test_results
            processed_files += 1
            
            logging.info(f"Successfully processed file: {filename}")
            logging.info(f"Number of samples: {len(test_results)}")
            logging.info(f"Tests per sample: {len(test_results[0])}")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

    if processed_files == 0:
        raise ValueError(f"No files were processed successfully! Examined files: {all_files}")

    if all_results is None:
        raise ValueError("No data was processed successfully!")
    
    # Create dataset with a single row containing all results
    dataset = Dataset.from_dict({
        'unit_tests_passed': [all_results]
    })
    
    # Final validation
    sample_data = dataset[0]['unit_tests_passed']
    true_count = sum(sum(1 for val in row if val) for row in sample_data)
    false_count = sum(sum(1 for val in row if not val) for row in sample_data)
    
    logging.info(f"Final dataset statistics:")
    logging.info(f"  - Number of rows in dataset: {len(dataset)}")
    logging.info(f"  - Total TRUE values: {true_count}")
    logging.info(f"  - Total FALSE values: {false_count}")
    
    if true_count == 0:
        raise ValueError("No TRUE values found in final dataset!")
        
    return dataset

def main(input_directory: str, output_filepath: str):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    dataset = process_directory(input_directory)
    
    logging.info(f"Saving dataset to {output_filepath}")
    dataset.save_to_disk(output_filepath)
    
    logging.info(f"Dataset saved successfully. Total samples: {len(dataset)}")

if __name__ == "__main__":
    save_dir = os.environ.get('SAVE_DIR')
    if not save_dir:
        raise ValueError("SAVE_DIR environment variable is not set")

    input_directory = os.path.join(save_dir, "eval_results", "cc_samples_Llama-3.1-8B-Instruct_1000_samples_50_unit_tests_MIDWAY_v4")
    output_filepath = os.path.join(save_dir, "good_turing", "cc_samples_Llama-3.1-8B-Instruct_1000_samples_50_unit_tests_MIDWAY_v4.hf")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    main(input_directory, output_filepath)