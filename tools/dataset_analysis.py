import argparse
import yaml
from Img2Vec.GroupNet.data.tokenizer import DevanagariTokenizer, MalayalamTokenizer, HindiTokenizer

def freq_counter(gt_file_path:str, charset:str, is_gt:bool = True):
    ngrp_freq = {}
    tokenizer = None

    if charset.lower() == 'devanagari':
        tokenizer= DevanagariTokenizer()
    elif charset.lower() == 'hindi':
        tokenizer= HindiTokenizer()
    elif charset.lower() == 'malayalam' :
        tokenizer = MalayalamTokenizer()
    else:
        raise NotImplementedError("Language Not implemented")
    
    char_frq = {c:0 for c in tokenizer.get_charset()}
    cntr = 1
    with open(gt_file_path, 'r') as f:
        for line in f:
            if is_gt:
                label = line.strip().split('\t')[1]
            else:
                label = line.strip()
            grps = tokenizer.label_transform(label)
            if len(grps) not in ngrp_freq.keys():
                ngrp_freq[len(grps)] = 1
            else:
                ngrp_freq[len(grps)] += 1

            for c in char_frq.keys():
                if c in label:
                    char_frq[c] += 1
            cntr += 1
            if cntr % 100000 == 0:
                print(f"Processes {cntr} number of lines")
    
    print("Character frequencies are:")
    for k,v in char_frq.items():
        print(f"{k}: {v}")

    print("Group Frequencies are:")
    for k,v in ngrp_freq.items():
        print(f"{k}: {v}")
        
if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Program to count the frequency of characters in Synthtiger gt.txt file')

    # Add arguments
    parser.add_argument('--gtfile', '-gt', type=str, help='gt.txt file path')
    parser.add_argument('--isgt', '-isgt', action='store_true', help='True if the file a synthtiger type GT file, else False')
    parser.add_argument('--charset','-c', type=str, help='Charset to count the freq in the labels\n \
                                                            "h" for hindi \n\
                                                            "m" for malayalam')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_file = args.gtfile
    is_gt = args.isgt
    charset = args.charset

    # Display the arguments
    print("Input file: ", input_file)
    print("charset: ", charset)
    print("isgt: ", is_gt)

    freq_counter(gt_file_path= input_file, charset= charset, is_gt=is_gt)
