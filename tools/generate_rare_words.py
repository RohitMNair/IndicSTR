import unicodedata
import random
from Img2Vec.GroupNet.data.tokenizer import MalayalamTokenizer
TOTAL_WORDS_PER_CHAR = 500000
low_freq_chars = {'ഖ', 'ഘ','ഛ','ഝ','ഠ','ഢ', 'ഥ', 'ഫ', 'ഈ',
                    'ഉ', 'ഋ','ഏ','ഐ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ',
                    'ഓ','ഔ','൦','൧','൨','൩','൪','൫','൬','൭', 'ൻ',
                    'ൾ', 'ൃ', 'ൈ', '൮','൯','₹','।','$', '!', '%',
                    'ൺ','ൗ', 'ഃ', '഻', 'ു്'}
low_freq_chars = set(unicodedata.normalize("NFKD", c) for c in low_freq_chars)
tokenizer = MalayalamTokenizer()

# with open("/home/rohitn/Vishnu/ank.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "w") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break
            
# with open("/home/rohitn/Vishnu/chihn.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/svar_matra.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/svar.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan_matra.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan_matra2.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break
            
# with open("/home/rohitn/Vishnu/vyanjan2_matra.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan2_matra2.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break
            
# with open("/home/rohitn/Vishnu/vyanjan2.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan3_matra.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan3_matra2.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan3.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

# with open("/home/rohitn/Vishnu/vyanjan4_matra.txt", 'r') as f_r, open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps.txt", "a") as f_a:
#     for line in f_r:
#         grp = unicodedata.normalize("NFKD",line.strip())
#         for char in low_freq_chars:
#             if char in grp:
#                 f_a.write(grp + '\n')
#                 break

grp_dict = {k:[] for k in low_freq_chars}
freq_dict = {k:0 for k in low_freq_chars}
with open("/home/rohitn/D1/MalayalamSynth/corpus/rare_grps_shuf.txt", 'r') as file:
    grps = file.readlines()
    i = 0
    while i < len(grps):
        grp = grps[i].strip()
        for k,v in grp_dict.items():
            if k in grp:
                grp_dict[k].append(grp)
        i += 1
        if i % 1000000 == 0:
            print("Finished ", i, " grps")

rare_words = set()
for k,v in grp_dict.items():
    if len(v) == 0:
        print(k,v)
    else:
        print(k, len(v))

while freq_dict:
    low_freq_chars_list = list(freq_dict.keys())
    # Randomly sample a number of items
    num_items_to_sample = random.randint(1, min(len(low_freq_chars_list), 10))
    # Randomly sample the character
    sampled_items = random.sample(low_freq_chars_list, num_items_to_sample)
    running_word = ''
    for char in sampled_items:
        if len(grp_dict[char]) == 0:
            print(char, sampled_items)
        running_word += random.sample(grp_dict[char], 1)[0]
    if len(tokenizer.label_transform(running_word)) != 0:
        for char in sampled_items:
            freq_dict[char] += 1
            if freq_dict[char] >= TOTAL_WORDS_PER_CHAR:
                del freq_dict[char]
            rare_words.add(running_word)

with open("/home/rohitn/D1/MalayalamSynth/corpus/rare_words.txt", "w") as file:
    for word in rare_words:
        file.write(word + "\n")

print("Success")