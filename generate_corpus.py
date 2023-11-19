"""
Code to generate the glyphs for devanagari script
the code will generate the following corpuses:-
1) svar
2) svar_matras
3) vyanjan_matras2
4) vyanjan_matra
5) vyanjan2
6) vyanjan
7) vyanjan2_matra
8) vyanjan2_matra2
9) ank
"""
# vyanjan
consonants = [
    'क', 'क़', 'ख', 'ख़', 'ग़', 'ज़', 'ग', 'घ', 'ङ', 'ड़', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट',
    'ठ', 'ड', 'ढ', 'ढ़', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'फ़',
    'ब', 'भ', 'म', 'य', 'य़', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 
    'स', 'ह', 'ऋ', 'ॠ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॾ', 'ॿ',
    ]
# ank
numerals = ['०','१','२','३','४','५','६' ,'७' ,'८' ,'९']
# svar
vowels = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऌ', 'ॡ', 'ऋ',
    'ॠ', 'ए', 'ऐ', 'ऎ', 'ऒ', 'ओ', 'औ', 'ॲ', 
    'ऍ', 'ऑ', 'ऄ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 
    ]
# matra
matras = [
    'ा','ि','ी','ु','ू','ृ','ॄ', 'ॢ', 'ॣ', 'े','ै', 'ॆ', 
    'ॊ', 'ो', 'ौ', 'ॏ', 'ऺ', '़', 'ॅ', 'ॉ', 'ँ','ं','ः',
     'ऻ', 'ॎ', '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', '॰', 'ॱ',
    ]
halfer = '्'
#chihn
punctuations = [
    '।', '!', '$', '₹', '%', '॥','ॽ', 
    ]

special_char = ['ऽ', 'ॐ', ]


def matra_sanity(m1:str, m2:str)->bool:
    """
    checks if 2 matras m1 & m2 are compatible or not
    """
    if m1 ==  'ः' and m2 not in ( 'ँ', 'ं', '॑', '॒', '॓', '॔', '॰', 'ॱ'):
        return False
    elif m1 == '्' and m2 not in ('़', 'ः', 'ँ'):
        return False
    elif m1 == 'ँ' and m2 != ':':
        return False
    elif m1 == 'ं' and m2 not in ('ॉ', 'ँ', 'ः',  '॑', '॒', '॓', '॰', 'ॱ'):
        return False
    elif m1 == '॑' and m2 not in ( '़', '्',  '॒', '॰', 'ॱ'):
        return False
    elif m1 == '॒' and m2 not in ('़', '्', '॑', '॰', 'ॱ'):
        return False
    elif (m1 == '॓' or m1 == '॔') and m2 not in ('़', '्',  'ँ', 'ं', 'ः', '॑','॰', 'ॱ'):
        return False
    elif m1 == '॰' or m1 == 'ॱ':
        return False
    else:
        return True

def svar_matra_sanity(v:str,m:str)->bool:
    """
    checks if matra is compatible with the svar
    """
    s = v + m
    bad = set((
        'अा','अऻ', 'अॆ', 'अॏ', 'अॊ','अो','अौ','अॅ','अॉ', 'अॖ', 'अॗ', 'आे','आै','आॆ',
        'आऺ', 'आॅ','उु','एे','एॆ','एॅ','अंा','अंि','अंी','अंु','अंू','अंृ',
        'अंॄ','अंॢ','अंॣ','अंे','अंै','अंॆ','अंॊ','अंो','अंौ','अं़',
        'अं:','अंॅ','अंॉ','अं्','अ:ा','अ:ि','अ:ी','अ:ु','अ:ू','अ:ृ','अ:ॄ',
        'अ:ॢ','अ:ॣ','अ:े','अ:ै','अ:ॆ','अ:ॊ','अ:ो','अ:ौ','अ:़', 'अंॏ', 'अंऺ'
        'अ:ॅ','अ:ॉ','अ:्','अ:ँ', 'अंऻ', 'अंॎ', 'अंॕ', 'अंॖ', 'अंॗ',
        'अ:ॏ','अ:ऺ','अ:ं','अ:ः','अ:ऻ','अ:ॎ','अ:॑','अ:॒','अ:॓', 'अ:॔',
        'अ:ॕ','अ:ॖ','अ:ॗ','अऺ'
        ))
    if s in bad:
        return False
    else:
        return True

def vyanjan_matra(save_directory)->None:
    """
    Function to produce corpus where there is a single vyanjan and single matra
    """
    with open(save_directory+ "vyanjan_matra.txt",'w',encoding='UTF-8') as f:
        for c in consonants:
            for m in matras:
                f.write(c + m +"\n")

def vyanjan(save_directory):
    """
    Function to generate corpus of just the vyanjans
    """
    with open(save_directory+ "vyanjan.txt",'w',encoding='UTF-8') as f:
        for c in consonants:
            f.write(c + "\n")
def svar(save_directory):
    """
    Function to generate corpus of just svar
    """
    with open(save_directory+ "svar.txt",'w',encoding='UTF-8') as f:
        for v in vowels:
            f.write(v + "\n")

def vyanjan_matra2(save_directory):
    """
    Function to generate corpus of a vyanjan with 2 matras
    """
    with open(save_directory+ "vyanjan_matra2.txt",'w',encoding='UTF-8') as f:
        for c in consonants:
            for m1 in matras:
                for m2 in matras:
                    if m1 != m2 and matra_sanity(m1,m2):
                        f.write(c + m1 + m2 + "\n")

def svar_matra(save_directory):
    """
    Function to generate corpus of a svar and a matra
    """
    with open(save_directory+ "svar_matra.txt",'w',encoding='UTF-8') as f:
        for v in vowels:
            for m in matras:
                if svar_matra_sanity(v,m):
                    f.write(v + m + "\n")

def vyanjan2(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full)
    """
    with open(save_directory+ "vyanjan2.txt",'w',encoding='UTF-8') as f:
        for c1 in consonants:
            for c2 in consonants:
                f.write(c1 + halfer + c2 + "\n")

def vyanjan2_matra(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full) with 1 matra
    """
    with open(save_directory+ "vyanjan2_matra.txt",'w',encoding='UTF-8') as f:
        for c1 in consonants:
            for c2 in consonants:
                for m in matras:
                    f.write(c1 + halfer + c2 + m + "\n")

def vyanjan2_matra2(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full) with 2 matras
    """
    with open(save_directory+ "vyanjan2_matra2.txt",'w',encoding='UTF-8') as f:
        for c1 in consonants:
            for c2 in consonants:
                for m1 in matras:
                    for m2 in matras:
                        if m1 != m2 and matra_sanity(m1,m2):
                            f.write(c1 + halfer + c2 + m1 + m2 + "\n")

def ank(save_directories):
    """
    Function to generate corpus of anks (digits)
    """
    with open(save_directory+ "ank.txt",'w',encoding='UTF-8') as f:
        for n in numerals:
            f.write(n + "\n")

def chihn(save_directory):
    """
    Function to generate corpus of chihn (punctuation) & special characters
    """
    with open(save_directory+ "chihn.txt",'w',encoding='UTF-8') as f:
        for n in special_char:
            f.write(n + "\n")

        for p in punctuations:
            f.write(p + "\n")


if __name__ == "__main__":
    save_directory = "../Corpus/"
    print(len(consonants), len(vowels), len(matras), len(punctuations))
    vyanjan_matra(save_directory)
    vyanjan(save_directory)
    svar(save_directory)
    vyanjan_matra2(save_directory)
    svar_matra(save_directory)
    vyanjan2(save_directory)
    vyanjan2_matra(save_directory)
    vyanjan2_matra2(save_directory)
    ank(save_directory)
    chihn(save_directory)
