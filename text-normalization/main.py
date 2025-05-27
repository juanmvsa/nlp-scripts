from text_preprocessing import read_txt_files
from normalize_text import normalize_documents
from pathlib import Path
from save_text_in_list import save_list_to_file
from remove_english_tokens import remove_english_tokens

folder_path = Path(
    "/Users/juanvasquez/jm.vsqz92@gmail.com - Google Drive/My Drive/chatbot-chris"
)

contents = read_txt_files(folder_path)
normalized_docs = normalize_documents(contents, correct_spanish_spelling=True)
final_docs = remove_english_tokens(normalized_docs)
save_list_to_file("pre_processed_text.txt", final_docs)
