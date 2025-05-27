from typing import List


def save_list_to_file(filename: str, text_list: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        for item in text_list:
            file.write(item + "\n")
