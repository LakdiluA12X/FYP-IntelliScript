import nltk
from collections import Counter
import os

root_folder = 'keyword-wise-filtered-docs'

leaf_folders = [os.path.join(root_folder, folder)
                for folder in os.listdir(root_folder)]


def join_word_pairs(words_list):
    joined_list = [words_list[i] + " " + words_list[i + 1] for i in range(0, len(words_list)) if i + 1 < len(words_list)
                   and words_list[i].isalnum() and words_list[i+1].isalnum()]

    # Handle the last element for odd-length lists (optional)
    if len(words_list) % 2 != 0:
        joined_list.append(words_list[-1])

    return joined_list


def calculate_filepath_keywords(folder_path):
    kw_list = []
    for root, directories, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(
                os.path.relpath(root, folder_path), filename)
            path_kw = file_path.split('/')
            processed_path_kw = []
            for w in path_kw:
                if w[-4:] == '.txt':
                    processed_path_kw.append(w[:-4])
                else:
                    processed_path_kw.append(w)

            kw_list.extend(processed_path_kw)

    return kw_list


# Download NLTK resources (run this only once)
nltk.download('punkt')
nltk.download('stopwords')

# List of stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add('page')
stop_words.add('number')
stop_words.add('http')


def count_words(folder_path):
    # Initialize counter to store keyword counts
    keyword_counts = Counter()
    keyword_2_counts = Counter()

    # Iterate through each file in the folder
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):  # Assuming files are text files
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().lower()  # Read file content and convert to lowercase
                    # Tokenize text into words
                    words = nltk.word_tokenize(text)

                    # Filter out stop words and non-significant words
                    keywords = [
                        word for word in words if word.isalnum() and word not in stop_words]
                    # Update keyword counts
                    keyword_counts.update(keywords)

                    # Combine adjacent items
                    words_2 = join_word_pairs(words)
                    keywords_2 = [
                        word for word in words_2 if word not in stop_words]
                    # Update keyword counts
                    keyword_2_counts.update(keywords_2)

    # Filter out words with count greater than 3
    filtered_keywords = {word: count for word,
                         count in keyword_counts.items() if count >= 2}
    filtered_keywords_2 = {word: count for word,
                           count in keyword_2_counts.items() if count >= 2}

    out_keywords = dict(sorted(dict(filtered_keywords).items(),
                        key=lambda item: item[1], reverse=True))
    out_keywords_2 = dict(sorted(
        dict(filtered_keywords_2).items(), key=lambda item: item[1], reverse=True))
    return out_keywords, out_keywords_2


def count_filename_words(folder_path):
    # Initialize counter to store keyword counts
    keyword_counts = Counter()

    words = calculate_filepath_keywords(folder_path)

    keywords = [word for word in words if word.isalnum()
                and word not in stop_words]
    keyword_counts.update(keywords)

    filtered_keywords = {word: count for word, count in keyword_counts.items()}
    out_keywords = dict(sorted(dict(filtered_keywords).items(),
                        key=lambda item: item[1], reverse=True))

    return out_keywords


for filepath in leaf_folders:
    words_json_1, words_json_2 = count_words(filepath)
    file_keywords = count_filename_words(filepath)
    # filename = '-'.join(str(filepath).split("\Extracted-text-data-v3\\")[1].split('\\')) + '.txt'
    filename = str(filepath).split("keyword-wise-filtered-docs/")[1] + '.txt'

    with open(os.path.join('filtered-docs-keywords', filename), 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(list(words_json_1.keys()) +
                      list(words_json_2.keys()) + list(file_keywords.keys())))
    print(filename)
