import zipfile
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(zip_file_path, text_file_name, test_size=0.1, random_state=42, encoding='ISO-8859-1'):
    texts = []
    labels = []
    try:
        # Use zipfile library to open the archive and the specific file
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(text_file_name, 'r') as file:
                for line in file:
                    # Decode line to string 
                    line = line.decode(encoding).strip()
                    parts = line.split('>')  # Assuming 'text>label' format
                    if len(parts) > 1:
                        texts.append(parts[0].strip())
                        labels.append(parts[1].strip())
        logging.info(f'Successfully read data from {text_file_name}')
    except Exception as e:
        logging.error(f'Error reading data: {e}')
        return [], [], [], []

    try:
        # Split the data into training and evaluation sets
        texts_train, texts_eval, labels_train, labels_eval = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        logging.info(f'Data split into training and evaluation sets with test_size={test_size}')
    except Exception as e:
        logging.error(f'Error splitting data: {e}')
        return [], [], [], []

    return texts_train, texts_eval, labels_train, labels_eval

# Specify the path to the zipped file
zip_file_path = '/content/drive/My Drive/WIKIFANE_gazet copy.txt.zip'
text_file_name = 'WIKIFANE_gazet copy.txt'  # Update the filename to match the file name inside the zip

# Call the function to load and split the data
texts_train, texts_eval, labels_train, labels_eval = load_and_split_data(zip_file_path, text_file_name)

# Print the sizes of the splits to verify it is correct
print(f"Training texts: {len(texts_train)}, Training labels: {len(labels_train)}")
print(f"Evaluation texts: {len(texts_eval)}, Evaluation labels: {len(labels_eval)}")
