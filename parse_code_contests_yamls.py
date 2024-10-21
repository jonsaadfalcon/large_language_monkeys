import os
import xml.etree.ElementTree as ET
import re
from datasets import Dataset
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_data(xml_string):
    breakpoint()
    
    root = ET.fromstring(xml_string)
    
    for document in root.findall('.//document'):
        content = document.find('document_content').text
        
        # Extract is_corrects
        is_corrects = re.findall(r'- (true|false)', content)
        is_corrects = [x == 'true' for x in is_corrects]
        
        # Extract unit_tests_passed
        unit_tests = re.findall(r'- (\d+\.\d+)', content)
        unit_tests = [float(x) for x in unit_tests]
        
        return {
            'is_corrects': is_corrects,
            'num_unit_tests_passed': len(unit_tests)
        }

def process_directory(directory_path):
    data = []
    
    logging.info(f"Processing files in directory: {directory_path}")
    xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]
    
    for filename in tqdm(xml_files, desc="Processing XML files"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        
        try:
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

    input_directory = os.path.join(save_dir, "eval_results", "pythia-2.8b_100_samples")
    output_filepath = os.path.join(save_dir, "good_turing", "pythia-2.8b_100_samples.hf")

    logging.info(f"SAVE_DIR is set to: {save_dir}")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output filepath: {output_filepath}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    logging.info(f"Ensured output directory exists: {os.path.dirname(output_filepath)}")

    main(input_directory, output_filepath)