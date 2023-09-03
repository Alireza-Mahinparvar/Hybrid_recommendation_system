import json

# Define a function to process AMiner-Author.txt
def process_author_file(input_filename, output_filename):
    authors = []

    with open(input_filename, 'r', encoding='utf-8') as file:
        author = {}
        for line in file:
            line = line.strip()
            if line:
                key_value = line.split(None, 1)
                if len(key_value) == 2:
                    key, value = key_value
                    key = key.strip('#')
                    author[key] = value
            else:
                authors.append(author)
                author = {}

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(authors, json_file, indent=4, ensure_ascii=False)

    print(f"Conversion complete. Data saved to '{output_filename}'")

# Define a function to process AMiner-Paper.txt
def process_paper_file(input_filename, output_filename):
    papers = []

    with open(input_filename, 'r', encoding='utf-8') as file:
        paper = {}
        for line in file:
            line = line.strip()
            if line:
                key_value = line.split(None, 1)
                if len(key_value) == 2:
                    key, value = key_value
                    key = key.strip('#')
                    if key == 'index':
                        if paper:
                            papers.append(paper)
                        paper = {}
                    if key in ('@', 'o', '%', '!'):
                        if key not in paper:
                            paper[key] = []
                        paper[key].extend(value.split(';'))
                    else:
                        paper[key] = value
            else:
                if paper:
                    papers.append(paper)
                paper = {}

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(papers, json_file, indent=4, ensure_ascii=False)

    print(f"Conversion complete. Data saved to '{output_filename}'")

# Define a function to process AMiner-Coauthor.txt
def process_coauthor_file(input_filename, output_filename):
    collaboration_data = {}

    with open(input_filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 3:
                    author1, author2, collaborations = parts
                    collaboration_data[(author1, author2)] = int(collaborations)

    collaboration_list = [{'author1': author1, 'author2': author2, 'collaborations': collaborations}
                          for (author1, author2), collaborations in collaboration_data.items()]

    with open(output_filename, 'w') as json_file:
        json.dump(collaboration_list, json_file, indent=4)

    print(f"Conversion complete. Data saved to '{output_filename}'")

# Define the input and output file names
input_filename = ''
output_filename = ''

# Check the input file name and perform the corresponding action
if 'AMiner-Author' in input_filename:
    process_author_file(input_filename, output_filename)
elif 'AMiner-Paper' in input_filename:
    process_paper_file(input_filename, output_filename)
elif 'AMiner-Coauthor' in input_filename:
    process_coauthor_file(input_filename, output_filename)
else:
    print(f"Unsupported file: {input_filename}. No action taken.")
