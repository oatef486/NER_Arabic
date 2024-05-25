import re
from collections import defaultdict

def extract_labels(file_path):
    label_count = defaultdict(int)
    # Open the file with 'utf-8-sig' to handle potential BOM issues
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        print(f"Opening file: {file_path}")  # Debugging step: print to confirm file is opened
        for line in file:
            print("Reading line:", line.strip())  # Debugging step: print to show each line read
            # Using a regex to find tags within the line
            labels = re.findall(r'<([A-Za-z_]+)>', line)
            if labels:
                print("Found labels:", labels)  # Debugging step: print to show found labels
            else:
                print("No labels found in line")  # Indicates regex did not match
            for label in labels:
                label_count[label] += 1
    return label_count


file_path = '/Users/omaratef/Desktop/NER/WIKIFANE_gazet.txt'
labels = extract_labels(file_path)
if not labels:  # Debugging step: Check if labels dictionary is empty
    print("No labels found.")
else:
    for label, count in labels.items():
        print(f"{label}: {count}")
