import os
import pandas as pd


def read_excel_file(filepath):
    # Read Excel file with multiple sheets
    xls = pd.ExcelFile(filepath)

    data = ''

    for sheet_name in xls.sheet_names:
        # Read data from the current sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)

        data += f'<new sheet>\nsheet_name: {sheet_name}\n'

        # Write data to the text file in comma format
        data += df.to_csv(sep=',', index=False,
                          header=False)+'\n'

    return data


def remove_consecutive_empty_lines(lines):
    # Initialize a counter to keep track of consecutive empty lines
    consecutive_empty_lines = 0

    # Iterate through the lines in the list
    i = 0
    while i < len(lines):
        # Check if the current line is empty
        if lines[i].strip() == '':
            # Increment the counter for consecutive empty lines
            consecutive_empty_lines += 1
            # Check if there are more than two consecutive empty lines
            if consecutive_empty_lines > 2:
                # Remove the current empty line from the list
                del lines[i]
                # Decrement the index to account for the removed line
                i -= 1
        else:
            # Reset the counter if the current line is not empty
            consecutive_empty_lines = 0
        # Move to the next line
        i += 1

    return lines


def clear_and_write_data(filepath, output_path):
    data = read_excel_file(filepath)
    data_lines = data.split('\n')
    modified_data = []
    for line in data_lines:
        if set(list(line)) != {'\r', ','}:
            modified_data.append(line)
        else:
            modified_data.append('')

    # Open a text file for writing
    with open(output_path, 'w', encoding='utf-8') as file:
        # Iterate over each sheet in the Excel file
        file.write('\n'.join(remove_consecutive_empty_lines(modified_data)))


source_dir = "cbsl-data-collector/CBSL-data"
output_dir = "excel-data-extractor/extracted-excel-files"
count = 0
error_count = 0


for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            if file.endswith('.xlsx') or file.endswith('.xls'):
                input_file_path = os.path.join(root, file)

                output_file_path = os.path.join(
                    output_dir, os.path.relpath(root, source_dir), file[:file.rfind('.')+1] + 'txt')

                clear_and_write_data(input_file_path, output_file_path)
                count += 1

        except Exception as e:
            print(e)
            error_count += 1

for root, dirs, files in os.walk(output_dir, topdown=False):
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder: {folder_path}")

print(f"Success: {count} - Errors: {error_count}")
