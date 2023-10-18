import functionfile
import os

file_path = functionfile.select_file()


#   Creates a list per row, all the list stored in a list
#   clean up symbols per Row
clean_Rows = functionfile.list_function(file_path, 'Job Description')
clean_Rows2 = functionfile.list_function(file_path, 'Position Level')
print('clean rows:', clean_Rows2)
clean_Rows2 = [item[0] for item in clean_Rows2]

#   Search for a keyword, then extract the 6 lines following the keyword.
extracted = functionfile.keyword_finder2(clean_Rows)

# Clean and format the extracted list: remove leading spaces and capitalize first letter
formatted_keywords = []
for item in extracted:
    if item:
        # Remove leading spaces and capitalize first letter of each line
        formatted_item = '\n'.join([line.strip().capitalize() for line in item.split('\n') if line.strip()])
        formatted_keywords.append(formatted_item)
    else:
        formatted_keywords.append('')


# Print the formatted keywords
print('\n\n'.join(formatted_keywords))

# Call the read_modify_combine_write function to include the modified columns
#   reads and writes to same file
#   change to desired location
print("****************" + file_path)
directory, base_file_with_extension = os.path.split(file_path)
base_file_name, extension = os.path.splitext(base_file_with_extension)
file_location = f"{base_file_name}_output{extension}"
output_file_path = os.path.join(directory, file_location)

functionfile.read_modify_combine_write(file_path, output_file_path, formatted_keywords, 'Job Description')
functionfile.read_modify_combine_write(output_file_path, output_file_path, clean_Rows2, 'Position Level')

# Remove the first occurrence of duplicate characters in the specified column
functionfile.remove_dupe(output_file_path, output_file_path, 'Company Location')

functionfile.remove_empty_rows(output_file_path, output_file_path)
