import os
from typing import List


def read_txt_files(folder_path: str) -> List[str]:
    """
    Reads all .txt files in the specified folder and returns their contents as strings in a list.

    Args:
        folder_path (str): Path to the folder containing the files.

    Returns:
        List[str]: List of strings, where each string contains the contents of a .txt file.
    """
    # List to store the contents of each .txt file
    txt_contents: List[str] = []

    try:
        # Get all files in the specified folder
        files: List[str] = os.listdir(folder_path)

        # Filter for only .txt files
        txt_files: List[str] = [file for file in files if file.endswith(".txt")]

        # Read each .txt file and add its contents to the list
        for txt_file in txt_files:
            file_path: str = os.path.join(folder_path, txt_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content: str = f.read()
                    txt_contents.append(content)
            except Exception as e:
                print(f"Error reading {txt_file}: {str(e)}")

        print(f"Total number of .txt files processed: {len(txt_contents)}")
        return txt_contents

    except Exception as e:
        print(f"Error accessing folder: {str(e)}")
        return txt_contents


# contents = read_txt_files(folder_path)

# # Print the number of files read
# print(f"\nnumber of .txt files read: {len(contents)}")

# # Optional: Preview the first few characters of each file
# for i, content in enumerate(contents):
#     preview = content[:50] + "..." if len(content) > 50 else content
#     print(f"\nfirst few lines of the file {i + 1} : {preview}")
