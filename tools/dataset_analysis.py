import argparse
from Img2Vec.GroupNet.data.tokenizer import Tokenizer, MalayalamTokenizer

hindi_half_character_classes = [ 'क', 'ख', 'ग', 'घ', 'ङ',
                            'च', 'छ', 'ज', 'झ', 'ञ',
                            'ट', 'ठ', 'ड', 'ढ', 'ण',
                            'त', 'थ', 'द', 'ध', 'न',
                            'प', 'फ', 'ब', 'भ', 'म',
                            'य', 'र', 'ल', 'ळ', 'व', 'श',
                            'ष', 'स', 'ह']
hindi_full_character_classes = ['अ', 'आ', 'इ', 'ई', 'उ', 
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
hindi_diacritic_classes= ['ा','ि','ी','ु','ू','ृ', 'े','ै', 'ो', 'ौ', 'ॅ', 'ॉ', 'ँ','ं', 'ः', '़'] # ensure that halfer is not in this 
hindi_halfer = '्'

mal_svar = ['അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'എ', 'ഏ','ഐ',
        'ഒ', 'ഓ', 'ഔ','അം','അഃ']  # 'ൠ' has not been added as it has not been used in recent malayalam
mal_vyanjan = ['ക', 'ഖ', 'ഗ', 'ഘ', 'ങ',
           'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ',
           'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ',
           'ത', 'ഥ', 'ദ', 'ധ', 'ന',
           'പ', 'ഫ', 'ബ', 'ഭ', 'മ',
           'യ', 'ര', 'ല', 'വ', 'ശ',
           'ഷ', 'സ', 'ഹ','ള','ഴ','റ'] # should add conditions to chillaksharam that it should not have any matra attached to it
mal_chillaksharam= ['ൺ','ൻ','ർ','ൽ','ൾ']
mal_chandrakala = '്'
mal_chihn = ['ഓം', '₹', '।', '!', '$', '%', '?','.',',',"-",'(',')'] # characters that occur independently and 'ൽ' has been taken out for confirmation
mal_ank = ['൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯']  # numbers
mal_matras = ['ാ','ി', 'ീ', 'ു', 'ൂ', 'ൃ','ൈ', 'ൊ', 'ോ', 'ൗ','ൌ', 'െ', 'േ', 'ം', 'ഃ', '഻']  # 'ു്' has been added because of its presence in older datasets added '഻' which sounds rr	 also has been added
mal_special_matra=['ം', 'ഃ']
mal_half_character_classes= mal_vyanjan
mal_full_character_classes= mal_svar+ mal_vyanjan+ mal_chillaksharam+ mal_ank + mal_chihn

def freq_counter(gt_file_path:str, charset:str, is_gt:bool = True):
    ngrp_freq = {}
    if charset == 'h':
        char_frq = {k:0 for k in (hindi_half_character_classes + hindi_full_character_classes + hindi_diacritic_classes + [hindi_halfer])}
        tokenizer = Tokenizer(
                        half_character_classes= hindi_half_character_classes,
                        full_character_classes= hindi_full_character_classes,
                        diacritic_classes= hindi_diacritic_classes,
                        halfer= hindi_halfer,
                    )
    elif charset == 'm':
        char_frq = {k:0 for k in (mal_half_character_classes + mal_full_character_classes + mal_matras + [mal_chandrakala])}
        tokenizer = MalayalamTokenizer(
            chill = mal_chillaksharam,
            special_matra= mal_special_matra,
            half_character_classes= mal_half_character_classes,
            full_character_classes = mal_full_character_classes,
            diacritic_classes= mal_matras,
            halfer= mal_chandrakala,
        )
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
