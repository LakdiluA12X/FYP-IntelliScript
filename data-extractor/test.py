import os
from extractor import Extractor


def get_file_paths(directory):
    file_paths = []
    # Walk through all files and directories in the given directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Create the full file path by joining the directory path and file name
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


# Example usage:
# directory_path = "test"
# files = get_file_paths(directory_path)

extractor = Extractor()

# for file in files:
#     print(file)

print(extractor.extractor('./test/PublicWaterMassMailing.pdf'))
