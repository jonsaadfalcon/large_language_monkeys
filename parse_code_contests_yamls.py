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
    Treats null values as false.
    """
    # Split on the marker
    splits = content.split("unit_tests_passed_individual_scores:")
    if len(splits) != 2:
        raise ValueError(f"Expected 1 occurrence of 'unit_tests_passed_individual_scores:', found {len(splits)-1}")
    
    # Get the relevant section and split into lines
    test_content = splits[1].strip()
    lines = [line.strip() for line in test_content.split('\n') if line.strip()]
    
    # Initialize containers
    matrix = []
    current_sample = []
    complete_true_count = 0
    complete_false_count = 0
    incomplete_true_count = 0
    incomplete_false_count = 0
    incomplete_samples = []
    
    # Log initial true count in raw content
    raw_true_count = test_content.count("true")
    logging.info(f"Raw 'true' count in content: {raw_true_count}")
    
    # Process each line
    for line_num, line in enumerate(lines):
        # Check if it's a new sample or continuation
        is_new_sample = line.startswith('- - ')
        
        # If it's a new sample and we have a complete previous sample
        if is_new_sample and current_sample:
            if len(current_sample) == 20:
                matrix.append(current_sample)
                complete_true_count += sum(1 for x in current_sample if x)
                complete_false_count += sum(1 for x in current_sample if not x)
            else:
                incomplete_samples.append(len(current_sample))
                incomplete_true_count += sum(1 for x in current_sample if x)
                incomplete_false_count += sum(1 for x in current_sample if not x)
            current_sample = []
            
        # Extract the value
        if is_new_sample:
            value = line[4:].strip()  # Remove '- - ' prefix
        else:
            value = line[2:].strip()  # Remove '- ' prefix
            
        # Convert to boolean and add to current sample
        current_sample.append(value == 'true')
            
    # Add the last sample if it exists
    if current_sample:
        if len(current_sample) == 20:
            matrix.append(current_sample)
            complete_true_count += sum(1 for x in current_sample if x)
            complete_false_count += sum(1 for x in current_sample if not x)
        else:
            incomplete_samples.append(len(current_sample))
            incomplete_true_count += sum(1 for x in current_sample if x)
            incomplete_false_count += sum(1 for x in current_sample if not x)
    
    # Log detailed statistics
    logging.info(f"Complete samples statistics:")
    logging.info(f"  - Number of complete samples: {len(matrix)}")
    logging.info(f"  - TRUE values in complete samples: {complete_true_count}")
    logging.info(f"  - FALSE values in complete samples: {complete_false_count}")
    
    logging.info(f"Incomplete samples statistics:")
    logging.info(f"  - Number of incomplete samples: {len(incomplete_samples)}")
    logging.info(f"  - TRUE values in incomplete samples: {incomplete_true_count}")
    logging.info(f"  - FALSE values in incomplete samples: {incomplete_false_count}")
    logging.info(f"  - Lengths of incomplete samples: {sorted(incomplete_samples)}")
    
    # Validation
    if not matrix:
        raise ValueError("No complete samples were parsed!")
        
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
    Returns a dataset with one row per file, each containing its test results.
    """
    logging.info(f"Processing files in directory: {directory_path}")
    
    # List all files in directory
    all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    logging.info(f"Found {len(all_files)} files in directory")
    
    # Initialize lists to store data for dataset
    all_results = []
    file_names = []
    processed_files = 0
    
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        logging.info(f"Attempting to process: {file_path}")
        
        try:
            test_results = process_file(file_path)
            
            # Add debugging information
            logging.info(f"File {filename} parsed results:")
            logging.info(f"  - Number of samples: {len(test_results)}")
            logging.info(f"  - True count in file: {sum(sum(1 for val in row if val) for row in test_results)}")
            
            # Add this file's results as a single row
            all_results.append(test_results)
            file_names.append(filename)
            processed_files += 1
            
            logging.info(f"Successfully processed file: {filename}")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

    if processed_files == 0:
        raise ValueError(f"No files were processed successfully! Examined files: {all_files}")

    if not all_results:
        raise ValueError("No data was processed successfully!")
    
    # Create dataset with one row per file
    dataset = Dataset.from_dict({
        'file_name': file_names,
        'unit_tests_passed': all_results
    })
    
    # Final validation with detailed logging
    total_true_count = sum(
        sum(sum(1 for val in row if val) for row in file_results)
        for file_results in dataset['unit_tests_passed']
    )
    total_false_count = sum(
        sum(sum(1 for val in row if not val) for row in file_results)
        for file_results in dataset['unit_tests_passed']
    )
    
    logging.info(f"Final dataset statistics:")
    logging.info(f"  - Number of rows in dataset: {len(dataset)}")
    logging.info(f"  - Total files processed: {processed_files}")
    logging.info(f"  - Total TRUE values: {total_true_count}")
    logging.info(f"  - Total FALSE values: {total_false_count}")
    logging.info(f"  - Average samples per file: {sum(len(x) for x in dataset['unit_tests_passed']) / len(dataset):.2f}")
    
    if total_true_count == 0:
        raise ValueError("No TRUE values found in final dataset!")
        
    return dataset

def main(input_directory: str, output_filepath: str):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    dataset = process_directory(input_directory)
    
    logging.info(f"Saving dataset to {output_filepath}")
    dataset.save_to_disk(output_filepath)
    
    logging.info(f"Dataset saved successfully. Total rows: {len(dataset)}")

if __name__ == "__main__":
    save_dir = os.environ.get('SAVE_DIR')
    if not save_dir:
        raise ValueError("SAVE_DIR environment variable is not set")

    folder_name = "cc_samples_Llama-3.1-8B-Instruct_1000_samples_50_unit_tests_v1"
    input_directory = os.path.join(save_dir, "eval_results", folder_name)
    output_filepath = os.path.join(save_dir, "good_turing", f"{folder_name}.hf")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    main(input_directory, output_filepath)