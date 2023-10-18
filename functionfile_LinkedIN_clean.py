import csv
from tkinter import Tk, filedialog


def list_function(file_path, column_name):
    """
    Read a specific column from a CSV file, removes symbols from to_replace[], and store each row in a list.

    Parameters:
        file_path (str): The path to the CSV file.
        column_name (str): The name of the column to read.

    Returns:
        list: A list of lists, where each sub-list corresponds to a row in the specified column.
    """
    data_list = []
    to_replace = ['â€¢', 'Â·        ', 'â€™', 'Â®', '  ', 'Â· ', '•', '·        ', '·        ',
                  '· ', '- ', '6. ', '7. ', '1. ', '2. ', '3. ', '4. ', '5. ', '·', '–']

    def replace_single_quote(string_with_single_quote):
        return string_with_single_quote.replace('’', '\'')

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned_item = row[column_name]
            for item in to_replace:
                cleaned_item = cleaned_item.replace(item, '')
            cleaned_item = cleaned_item.lower()
            cleaned_item = replace_single_quote(cleaned_item)
            data_list.append([cleaned_item])

    return data_list


def keyword_finder2(column_data, num_lines=6):
    extracted_data = []

    # Iterate through each item in the column_data
    for item in column_data:
        if len(item) > 0:
            text = item[0]  # Get the first element of the sublist
            if isinstance(text, str):  # Check if it's a string
                lines = text.split('\n')  # Split the text into lines

                #   Find the index where "requirements" is mentioned
                #   add on more keywords if needed
                req_index = -1
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in
                           ['requirements', 'required', 'What we are looking for']):
                        req_index = i
                        break

                if req_index != -1:
                    # Extract the following num_lines lines after "requirements"
                    extracted_text = "\n".join(lines[req_index + 1:req_index + num_lines + 1])
                    extracted_data.append(extracted_text)
                else:
                    extracted_data.append(None)
            else:
                extracted_data.append(None)
        else:
            extracted_data.append(None)

    if all(extracted is None for extracted in extracted_data):
        print(f'Keywords {"requirements", "required"} not found in any row.')

    return extracted_data


def read_modify_combine_write(input_file_path, output_file_path, modified_column_data, column_name):
    """
    Read a CSV file, modify a specified column with provided data, and write the modified data to a new CSV file.

    Parameters:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str): The path to the output CSV file.
        modified_column_data (list): A list of modified data for the specified column.
        column_name (str): The name of the column to modify.

    Returns:
        None
    """
    # Read the CSV file and store the data in a list of dictionaries

    data = []
    with open(input_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    # Add the modified column data to the data
    for i, row in enumerate(data):
        if column_name in row:
            row[column_name] = modified_column_data[i] if i < len(modified_column_data) else ''

    # Write the modified data to a new CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write each modified row to a new row
        for row in data:
            writer.writerow(row)


#   testing
def remove_dupe(input_file, output_file, column_name):
    """
    Remove the first occurrence of duplicate characters in each row of a specified column in a CSV file.

    Parameters:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.
        column_name (str): The name of the column to process.

    Returns:
        None
    """
    # Read the input CSV file and process the specified column
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        # Create a new list of rows with processed column
        rows = []
        for row in reader:
            if column_name in row:
                original_value = row[column_name]
                processed_value = remove_duplicates(original_value)
                row[column_name] = processed_value

            rows.append(row)

    # Write the modified data to the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_duplicates(value):
    """
    Remove one occurrence of the pattern "Singapore, " followed by "Singapore" and subsequent commas in a string.

    Parameters:
        value (str): The input string.

    Returns:
        str: The modified string with one occurrence of the pattern "Singapore, "
        followed by "Singapore" and subsequent commas removed.
    """
    # Find the position of the first occurrence of "Singapore, "
    start_index = value.find('Singapore, ')

    if start_index != -1:
        # Find the position of the next occurrence of "Singapore" after the first occurrence of "Singapore, "
        end_index = value.find('Singapore', start_index + len('Singapore, '))

        # Remove one occurrence of "Singapore, " followed by "Singapore" and subsequent commas if found
        if end_index != -1:
            modified_value = value[:start_index] + value[end_index:].replace('Singapore, ', '', 1)
            return modified_value

    return value


def remove_empty_rows(input_file, output_file):
    """
    Read a CSV file, check a specific column, and remove entire rows where that column is empty.

    Parameters:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.

    Returns:
        None
    """
    # Read the input CSV file and remove empty rows
    rows_to_keep = []

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Check if column E 'Job Description' is not empty
            if row and row[4]:
                rows_to_keep.append(row)

    # Write the modified data to the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows_to_keep)


# Function to prompt the user to select a file using a file dialog
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    return file_path
