import os
import re

source_dir = '../Extracted-text-data-v3'
keywords = ['interest rate', 'inflation rate', 'economic growth rate', 'prosperity index',
            'bank rate', 'unemployment', 'exchange rate', 'monetary policy', 'laws']

file_counts = {}

for kw in keywords:
    os.makedirs(kw, exist_ok=True)
    file_counts[kw] = 0

for kw in keywords:
    for root, directories, files in os.walk(source_dir):
        for directory in directories:
            # Create corresponding directories in the output directory
            output_subdir = os.path.join(
                kw, os.path.relpath(root, source_dir), directory)
            os.makedirs(output_subdir, exist_ok=True)

for root, directories, files in os.walk(source_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Read file content and convert to lowercase

        for topic in keywords:
            if topic in text:
                pages = [j[j.find('>')+1:] for j in text.split('<page number')]
                temp_text = ''
                for page in pages:
                    if topic in page:
                        lines = page.split('\n')
                        m = 4
                        for i in range(m, len(lines)-m):
                            if topic in lines[i]:
                                text_to_check = '\n'.join(lines[i-m:i+m])
                                match = bool(re.search(r'\d', text_to_check))
                                if match:
                                    temp_text += page
                                    temp_text += '\n<page sep>\n'
                                    break

                output_file_path = os.path.join(
                    topic, os.path.relpath(root, source_dir), filename)

                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(temp_text)

                file_counts[topic] += 1

for kw in keywords:
    for root, dirs, files in os.walk(kw, topdown=False):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            if not os.listdir(folder_path):  # Check if folder is empty
                os.rmdir(folder_path)
                print(f"Deleted empty folder: {folder_path}")

print(file_counts)
