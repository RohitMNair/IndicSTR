from Img2Vec.GroupNet.data.tokenizer import MalayalamTokenizer, HindiTokenizer
import unicodedata
import random

RARE_GROUP_NOS = [1,2,3]
corpus = []
tokenizer = MalayalamTokenizer()
with open("/home/rohitn/D1/MalayalamSynth/corpus/malayalam_words_uniq.txt", 'r') as file:
    corpus = file.readlines()

grp_freq = {no:0 for no in RARE_GROUP_NOS}
words = []
while grp_freq:
    for no in grp_freq.keys():
        sampled_word = None
        while True:
            sampled_word = tokenizer.label_transform(
                                unicodedata.normalize(
                                    "NFKD",
                                    random.sample(corpus, 1)[0].strip(),
                                    )
                            )
            if len(sampled_word) > no:
                break
        
        running_word = tokenizer.label_transform(''.join(random.sample(sampled_word, no)))
        if len(running_word) > 0:
            words.append(''.join(running_word))
            grp_freq[no] += 1

    if grp_freq[no] > 500000:
        del grp_freq[no]
    
    if len(words) % 10000 == 0:
        print(f"Generated {len(words)}")

with open("/home/rohitn/D1/MalayalamSynth/corpus/rare_group_nos.txt", "w") as f:
    for word in words:
        f.write(word + '\n')

print("SUCCESS!!")