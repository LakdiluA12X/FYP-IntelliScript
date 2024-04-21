import os
import shutil

source_dir = "excel-data-extractor/Extracted-text-data/"
output_dir = "excel-data-extractor/Extracted-text-data-cleaned/"
count = 0
error_count = 0
empty_files = 0


for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            with open(input_file_path, 'r', encoding='utf-8') as infile:
                data = infile.read()

            # Determine the output file path
            output_file_path = os.path.join(
                output_dir, os.path.relpath(root, source_dir), file[:-3] + 'txt')

            if data != '' and len(data) > 500:
                shutil.copy(input_file_path, output_file_path)
                count += 1

            else:
                empty_files += 1

        except Exception as e:
            error_count += 1
            print(f"{error_count}E - {e} - {input_file_path}")

for root, dirs, files in os.walk(output_dir, topdown=False):
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder: {folder_path}")

print(f"Success: {count} - Errors: {error_count} - Empty: {empty_files}")
