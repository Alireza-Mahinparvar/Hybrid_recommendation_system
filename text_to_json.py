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
# Create an empty list to store paper records
papers = []

# Read the input file and parse the records with UTF-8 encoding
with open(input_filename, 'r', encoding='utf-8') as file:
    paper = {}  # Initialize an empty paper record
    for line in file:
        line = line.strip()
        if line:
            key_value = line.split(None, 1)
            if len(key_value) == 2:
                key, value = key_value
                key = key.strip('#')  # Remove '#' character from key
                if key == 'index':
                    # If the key is 'index', it's the start of a new paper record
                    if paper:
                        papers.append(paper)
                    paper = {}  # Initialize a new paper record
                    # Map 'index' to 'id'
                    paper['id'] = value
                elif key == '*':
                    # Map '*' to 'paper title'
                    paper['paper title'] = value
                elif key == '@':
                    # Map '@' to 'authors'
                    paper['authors'] = value.split(';')
                elif key == 'o':
                    # Map 'o' to 'affiliations'
                    paper['affiliations'] = value.split(';')
                elif key == 't':
                    # Map 't' to 'year'
                    paper['year'] = value
                elif key == 'c':
                    # Map 'c' to 'publication venue'
                    paper['publication venue'] = value
                elif key == '%':
                    # Map '%' to 'references'
                    if 'references' not in paper:
                        paper['references'] = []
                    paper['references'].append(value)
                elif key == '!':
                    # Map '!' to 'abstract'
                    paper['abstract'] = value
        else:
            # Blank line indicates the end of a paper record
            if paper:
                papers.append(paper)
            paper = {}

# Write the parsed data to a JSON file
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
