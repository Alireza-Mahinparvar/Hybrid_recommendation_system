import json
import re

# Define a function to process AMiner-Author.txt and save as JSON
def process_author_file(input_filename, output_filename):
    authors = []

    with open(input_filename, 'r', encoding='utf-8', errors='ignore') as file:
        author = {}
        for line in file:
            line = line.strip()
            if line.startswith("#index"):
                author['authorid'] = line[7:].strip()
            elif line.startswith("#n"):
                author['name'] = line[3:].strip()
            elif line.startswith("#a"):
                author['aff'] = re.sub(r'[^\w.]', '', line[2:].strip())
            elif line.startswith("#pc"):
                author['pc'] = line[4:].strip()
            elif line.startswith("#cn"):
                author['cn'] = line[4:].strip()
            elif line.startswith("#hi"):
                author['hi'] = line[4:].strip()
            elif line.startswith("#pi"):
                author['pi'] = line[4:].strip()
            elif line.startswith("#upi"):
                author['upi'] = line[5:].strip()
            elif line.startswith("#t"):
                author['interest'] = line[3:].strip()
            elif not line:
                authors.append(author)
                author = {}

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(authors, json_file, indent=4, ensure_ascii=False)

    print(f"Conversion complete. Data saved to '{output_filename}'")

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
input_author_filename = 'AMiner-Author.txt'
output_author_filename = 'AMiner-Author.json'

input_paper_filename = 'AMiner-Paper.txt'
output_paper_filename = 'AMiner-Paper.json'

input_coauthor_filename = 'AMiner-Coauthor.txt'
output_coauthor_filename = 'AMiner-Coauthor.json'

# Process and save data as JSON
process_author_file(input_author_filename, output_author_filename)
process_paper_file(input_paper_filename, output_paper_filename)
process_coauthor_file(input_coauthor_filename, output_coauthor_filename)
