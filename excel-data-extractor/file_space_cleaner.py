import os


source_dir = "excel-data-extractor/Extracted-text-data-cleaned"
output_dir = "excel-data-extractor/Extracted-text-data-space-cleaned"

count = 0
error_count = 0


def count_starting_spaces(line):
    """Count the number of starting spaces in a line."""
    return len(line) - len(line.lstrip())


def keep_two_consecutive_blank_lines(lines):
    result = []
    consecutive_blank_lines = 0

    for line in lines:
        if line.strip() == "":
            consecutive_blank_lines += 1
            if consecutive_blank_lines <= 2:
                result.append(line)
        else:
            result.append(line)
            consecutive_blank_lines = 0

    return result


for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            # Determine the output file path
            output_file_path = os.path.join(
                output_dir, os.path.relpath(root, source_dir), file[:file.rfind('.')+1] + 'txt')

            with open(input_file_path, 'r', encoding='utf-8') as infile:
                datalines = infile.read().split('\n')

            space_count = []
            for line in datalines:
                if line != '' and line[0] != '<':
                    space_count.append(count_starting_spaces(line))

            cleaned_data = []
            remove_spaces = min(space_count)
            for line_ in datalines:
                if line_ != '' and line_[0] != '<':
                    cleaned_data.append(line_[remove_spaces:].rstrip())

                else:
                    cleaned_data.append(line_.rstrip())

            new_data = keep_two_consecutive_blank_lines(cleaned_data)

            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write('\n'.join(new_data))

            count += 1

        except Exception as e:
            error_count += 1
            print(f"{error_count}E - {e} - {input_file_path}")

for root, dirs, files in os.walk(output_dir, topdown=False):
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder: {folder_path}")

print(f"Success: {count} - Errors: {error_count}")
