import os
import xml.etree.ElementTree as ET
from datasets import Dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_boolean_list(content_str: str) -> List[bool]:
    """Parse the boolean list from the content string."""
    # Split the string into lines and filter out empty lines
    lines = [line.strip() for line in content_str.split('\n') if line.strip()]
    
    # Extract boolean values
    bools = []
    for line in lines:
        if isinstance(line, str):
            parts = line.strip().split('-')
            if len(parts) > 1:
                value = parts[-1].strip()
                bools.append(value == 'true')
    
    return bools

def extract_data(xml_content: str) -> Dict[str, Any]:
    """Extract data from XML content."""
    root = ET.fromstring(xml_content)
    
    # Find the document with the content
    document = root.find(".//document_content")
    if document is None:
        raise ValueError("No document content found in XML")
    
    content = document.text.strip()
    
    # Split content into lines and process
    lines = content.split('\n')
    
    # Process each row
    all_rows = []
    current_row = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('  - '):
            # Values in a row
            value = line[4:].strip()
            current_row.append(value == 'true')
        elif line.startswith('- '):
            # New row
            if current_row:
                if len(current_row) == 20:  # Only add complete rows
                    all_rows.append(current_row)
                current_row = []
    
    # Add the last row if it exists and is complete
    if current_row and len(current_row) == 20:
        all_rows.append(current_row)
    
    if not all_rows:
        raise ValueError("No valid rows found in the content")
    
    return {
        'tests_matrix': all_rows,
        'num_rows': len(all_rows),
        'num_cols': 20  # We know this should always be 20
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
                
            # Check if content contains XML structure
            if '<documents>' in content and '</documents>' in content:
                result = extract_data(content)
                result['filename'] = filename
                data.append(result)
            else:
                logging.warning(f"Skipping {filename}: Not in expected XML format")
                
        except ET.ParseError as e:
            logging.error(f"XML parsing error in {filename}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
    
    if not data:
        logging.error("No valid files were processed!")
    else:
        logging.info(f"Successfully processed {len(data)} files")
    
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