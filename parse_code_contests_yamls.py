import os
import xml.etree.ElementTree as ET
import re
from datasets import Dataset
from tqdm import tqdm

def extract_data(xml_string):
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
    
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                xml_content = file.read()
            
            try:
                result = extract_data(xml_content)
                result['filename'] = filename
                data.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return data

def create_dataset(data):
    return Dataset.from_list(data)

def main(input_directory, output_filepath):
    print("Processing XML files...")
    data = process_directory(input_directory)
    
    print("Creating dataset...")
    dataset = create_dataset(data)
    
    print("Saving dataset...")
    dataset.save_to_disk(output_filepath)
    
    print(f"Dataset saved to {output_filepath}")
    print(f"Total samples processed: {len(dataset)}")

if __name__ == "__main__":
    save_dir = os.environ.get('SAVE_DIR')
    if not save_dir:
        raise ValueError("SAVE_DIR environment variable is not set")

    input_directory = os.path.join(save_dir, "eval_results", "pythia-2.8b_100_samples")
    output_filepath = os.path.join(save_dir, "good_turing", "pythia-2.8b_100_samples.hf")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    main(input_directory, output_filepath)