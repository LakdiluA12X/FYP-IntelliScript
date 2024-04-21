import os
import shutil

# Paths to the two folder structures
error_files = 'excel-data-extractor/Extracted-text-data-space-cleaned'
correct_files = 'excel-data-extractor/extracted-excel-files'
count = 0
errors = 0

# Iterate over the text files in folder structure 2
for root, dirs, files in os.walk(correct_files):
    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            with open(input_file_path, 'r', encoding='utf-8') as infile:
                correct_data = infile.read()

            output_file_path = os.path.join(
                error_files, os.path.relpath(root, correct_files), file[:file.rfind('.')+1] + 'txt')

            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(correct_data)

            count += 1

        except:
            errors += 1


print(f"Success: {count} - Errors: {errors}")
