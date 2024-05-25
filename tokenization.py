from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.ner import NERecognizer

# Function to read data from a file
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# The function to preprocess text
def preprocess_text(text):
    # Normalize text by replacing underscores with spaces
    text = text.replace('_', ' ')
    return text

# Initialize the NER model
ner = NERecognizer.pretrained()


file_path = '/Users/omaratef/Desktop/NER/WikiFANEGazet/WIKIFANE_gazet.txt'  

raw_text = read_data(file_path)
processed_text = preprocess_text(raw_text)

# Tokenize the text
tokens = simple_word_tokenize(processed_text)

# Predict entities using Camel Tools NER
entities = ner.predict(tokens)

# Output the results
for token, entity in zip(tokens, entities):
    print(f"{token}: {entity}")
