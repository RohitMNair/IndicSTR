import argparse
from Img2Vec.GroupNet.data.tokenizer import Tokenizer

half_character_classes = [ 'क', 'ख', 'ग', 'घ', 'ङ',
                            'च', 'छ', 'ज', 'झ', 'ञ',
                            'ट', 'ठ', 'ड', 'ढ', 'ण',
                            'त', 'थ', 'द', 'ध', 'न',
                            'प', 'फ', 'ब', 'भ', 'म',
                            'य', 'र', 'ल', 'ळ', 'व', 'श',
                            'ष', 'स', 'ह']
full_character_classes = ['अ', 'आ', 'इ', 'ई', 'उ', 
                'ऊ', 'ऋ', 'ॠ', 'ए', 'ऐ',
                'ओ', 'औ', 'ॲ', 'ऍ', 'ऑ',
                'क', 'ख', 'ग', 'घ', 'ङ',
                'च', 'छ', 'ज', 'झ', 'ञ',
                'ट', 'ठ', 'ड', 'ढ', 'ण',
                'त', 'थ', 'द', 'ध', 'न',
                'प', 'फ', 'ब', 'भ', 'म',
                'य', 'र', 'ल', 'ळ', 'व', 
                'श', 'ष', 'स', 'ह',
                'ॐ', '₹', '।', '!', '$', ',', '.', '-', '%', '॥','ॽ', # characters that occur independently
                '०','१','२','३','४','५','६' ,'७' ,'८' ,'९', # numerals
                ]
diacritic_classes= ['ा','ि','ी','ु','ू','ृ', 'े','ै', 'ो', 'ौ', 'ॅ', 'ॉ', 'ँ','ं', 'ः', '़'] # ensure that halfer is not in this 
halfer = '्'

def freq_counter(gt_file_path:str, charset:str):
    char_frq = {k:0 for k in (half_character_classes + full_character_classes + diacritic_classes + [halfer])}
    ngrp_freq = {}
    tokenizer = Tokenizer(
                    half_character_classes= half_character_classes,
                    full_character_classes= full_character_classes,
                    diacritic_classes= diacritic_classes,
                    halfer= halfer,
                    max_grps = 25,
                )
    cntr = 1
    with open(gt_file_path, 'r') as f:
        for line in f:
            label = line.strip().split('\t')[1]
            grps = tokenizer.hindi_label_transform(label)
            if len(grps) not in ngrp_freq.keys():
                ngrp_freq[len(grps)] = 1
            else:
                ngrp_freq[len(grps)] += 1

            for c in label:
                if c in char_frq.keys():
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
    parser.add_argument('--charset','-c', type=str, help='Charset to count the freq in the labels\n \
                                                            "h" for hindi \n\
                                                            "m" for malayalam')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_file = args.gtfile
    charset = args.charset

    # Display the arguments
    print("Input file: ", input_file)
    print("charset: ", charset)

    freq_counter(input_file, charset)
