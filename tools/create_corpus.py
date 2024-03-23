from Img2Vec.GroupNet.data.tokenizer import DevanagariTokenizer, MalayalamTokenizer, HindiTokenizer
from typing import Union
import unicodedata
import yaml
import argparse

def select_tokenizer(charset:str)->Union[DevanagariTokenizer, MalayalamTokenizer, HindiTokenizer]:
    """
    Returns an instantiated tokenizer for the language
    """
    if charset == 'devanagari':
        return DevanagariTokenizer()
    elif charset == 'hindi':
        return HindiTokenizer()
    elif charset == 'malayalam':
        return MalayalamTokenizer()
    else:
        raise NotImplementedError("Language hasn't been implemented yet")
        
def extract_unique_words(input_file:str, output_file:str,\
                          tokenizer:Union[DevanagariTokenizer, MalayalamTokenizer])-> None:
    """
    Indentifies unique words in the corpus
    Args:
    - input_file: string path of file containing text or words of a language
    - output_file: string path of the output file
    - tokenizer: Instance of a language tokenizer for parsing the words
    """
    # Read the input text file
    print("Reading File")
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # split the sentences by whitespaces the text into words
    words = text.split()

    # Get unique words
    print("Identifying Unique words")
    unique_words = set(unicodedata.normalize('NFKD', word) for word in words)

    print(f"Number of Unique Words:{len(unique_words)} \nFiltering words using Tokenizer...")
    # filter the words according to the tokenizer
    filtered_words = set()
    charset = tokenizer.get_charset()
    for word in unique_words:
        word_copy = ''.join(c for c in word if c in charset)
        if len(tokenizer.label_transform(word_copy)) != 0:
            filtered_words.add(word_copy)

    print(f"Writing words to file {output_file}")
    with open(output_file, 'w') as of:
        for word in filtered_words:
            of.write(word + '\n')

    print("Words written!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("--input_file", "-inp", type=str, help="Path to the input text file.")
    parser.add_argument("--output_file", "-out", type=str, help="Path to the output file.")
    parser.add_argument("--charset", "-c", type=str, help="name of the character set to use.")

    args = parser.parse_args()

    input_filename = args.input_file  # Replace with the path to your input file
    output_filename = args.output_file  # Replace with the desired output file name
    charset = args.charset

    extract_unique_words(input_file= input_filename,output_file= output_filename, tokenizer= \
                         select_tokenizer(charset= charset))