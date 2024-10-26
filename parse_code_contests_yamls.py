import os
import xml.etree.ElementTree as ET
from datasets import Dataset
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_boolean_list(content_str):
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

def extract_data(xml_content):
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
        if line.startswith('- '):
            # Start of a new value
            value = line[2:].strip()
            current_row.append(value == 'true')
        elif line == '-':
            # Empty row marker
            if current_row:
                all_rows.append(current_row)
                current_row = []
        elif line and line[0] == '-':
            # Start of a new row
            if current_row:
                all_rows.append(current_row)
            current_row = []
    
    # Add the last row if it exists
    if current_row:
        all_rows.append(current_row)
    
    # Filter out empty rows and rows that don't have exactly 20 elements
    valid_rows = [row for row in all_rows if len(row) == 20]
    
    return {
        'tests_matrix': valid_rows
    }

def process_directory(directory_path):
    data = []
    
    logging.info(f"Processing files in directory: {directory_path}")
    xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]
    
    for filename in tqdm(xml_files, desc="Processing XML files"):
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                xml_content = file.read()
            
            result = extract_data(xml_content)
            result['filename'] = filename
            data.append(result)
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
    
    logging.info(f"Processed {len(data)} files successfully")
    return data

def create_dataset(data):
    logging.info("Creating Hugging Face dataset")
    return Dataset.from_list(data)

def main(input_directory, output_filepath):
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_filepath}")

    data = process_directory(input_directory)
    
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