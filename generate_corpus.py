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
10) Chihn
"""
# # vyanjan
# consonants = [
#     'क', 'क़', 'ख', 'ख़', 'ग़', 'ज़', 'ग', 'घ', 'ङ', 'ड़', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट',
#     'ठ', 'ड', 'ढ', 'ढ़', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'फ़',
#     'ब', 'भ', 'म', 'य', 'य़', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 
#     'स', 'ह', 'ऋ', 'ॠ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॾ', 'ॿ',
#     ]
# # ank
# numerals = ['०','१','२','३','४','५','६' ,'७' ,'८' ,'९']
# # svar
# vowels = [
#     'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऌ', 'ॡ', 'ऋ',
#     'ॠ', 'ए', 'ऐ', 'ऎ', 'ऒ', 'ओ', 'औ', 'ॲ', 
#     'ऍ', 'ऑ', 'ऄ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 
#     ]
# # matra
# devanagari_matras = [
#     'ा','ि','ी','ु','ू','ृ','ॄ', 'ॢ', 'ॣ', 'े','ै', 'ॆ', 
#     'ॊ', 'ो', 'ौ', 'ॏ', 'ऺ', '़', 'ॅ', 'ॉ', 'ँ','ं','ः',
#      'ऻ', 'ॎ', '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', '॰', 'ॱ',
#     ]

# halfer = '्'
# #chihn
# punctuations = [
#     '।', '!', '$', '₹', '%', '॥','ॽ', 
#     ]

# special_char = ['ऽ', 'ॐ']

# iiit matras = ['ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ']
# iiit characters = {'अ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च', 'छ',
#  'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ',
#  'म', 'य', 'र', 'ऱ', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ॐ', '०', '१', '२', '३', '४', '५', '७', '९'}
# MLT Characters = {'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख',
#  'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ',
#  'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ॐ', 'ड़', 'ढ़', '०', '१', '२', '३', '४', '५',
#  '६', '७', '८', '९'}
# MLT Matras = {'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ'}
# MLT Characters not included in this set but will be handled sepearately
# {'ड़', 'ढ़'} 
# IIIT characters not included but will be handled seperately= {'ऱ'}
svar = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ए', 'ऐ',
        'ओ', 'औ', 'ॲ', 'ऍ', 'ऑ']
vyanjan = ['क', 'ख', 'ग', 'घ', 'ङ',
           'च', 'छ', 'ज', 'झ', 'ञ',
           'ट', 'ठ', 'ड', 'ढ', 'ण',
           'त', 'थ', 'द', 'ध', 'न',
           'प', 'फ', 'ब', 'भ', 'म',
           'य', 'र', 'ल', 'ळ', 'व', 'श',
           'ष', 'स', 'ह',
           #'क़', 'ख़', 'ग़', 'ज़', 'झ़', 'ड़', 'ढ़', 'फ़' # these will be handled by the matra '़'
           # For real text, each of these will be converted to vyanjan + matra
           ]
halanth = '्'
chihn = ['ॐ', '₹', '।', '!', '$', '₹', '%', '॥','ॽ'] # characters that occur independently
ank = ['०','१','२','३','४','५','६' ,'७' ,'८' ,'९'] # numbers
matras = ['ा','ि','ी','ु','ू','ृ', 'े','ै', 'ो', 'ौ', 'ॅ', 'ॉ', 'ँ','ं', 'ः', '़'] # added nuktha

def matra_sanity(m1:str, m2:str)->bool:
    """
    checks if 2 matras m1 & m2 are compatible or not
    """
    if m1 ==  'ः' and m2 not in ( 'ँ', 'ं', '॑', '॒', '॓', '॔', '॰', 'ॱ'):
        return False
    elif m1 == '्' and m2 not in ('़', 'ः', 'ँ'):
        return False
    elif m1 == 'ँ' and m2 != 'ः':
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

def svar_matra_sanity(s:str,m:str)->bool:
    """
    checks if matra is compatible with the svar
    """
    s_m = s + m
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
    if s_m in bad:
        return False
    else:
        return True

def vyanjan_matra(save_directory)->None:
    """
    Function to produce corpus where there is a single vyanjan and single matra
    """
    with open(save_directory+ "vyanjan_matra.txt",'w',encoding='UTF-8') as f:
        for c in vyanjan:
            for m in matras:
                f.write(c + m +"\n")

def vyanjan_only(save_directory)->None:
    """
    Function to generate corpus of just the vyanjans
    """
    with open(save_directory+ "vyanjan.txt",'w',encoding='UTF-8') as f:
        for c in vyanjan:
            f.write(c + "\n")
def svar_only(save_directory):
    """
    Function to generate corpus of just svar
    """
    with open(save_directory+ "svar.txt",'w',encoding='UTF-8') as f:
        for s in svar:
            f.write(s + "\n")

def vyanjan_matra2(save_directory):
    """
    Function to generate corpus of a vyanjan with 2 matras
    """
    with open(save_directory+ "vyanjan_matra2.txt",'w',encoding='UTF-8') as f:
        for c in vyanjan:
            for m1 in matras:
                for m2 in matras:
                    if m1 != m2 and matra_sanity(m1,m2):
                        f.write(c + m1 + m2 + "\n")

def svar_matra(save_directory):
    """
    Function to generate corpus of a svar and a matra
    """
    with open(save_directory+ "svar_matra.txt",'w',encoding='UTF-8') as f:
        for s in svar:
            for m in matras:
                if svar_matra_sanity(s,m):
                    f.write(s + m + "\n")

def vyanjan2(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full)
    """
    with open(save_directory+ "vyanjan2.txt",'w',encoding='UTF-8') as f:
        for v1 in vyanjan:
            for v2 in vyanjan:
                f.write(v1 + halanth + v2 + "\n")

def vyanjan2_matra(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full) with 1 matra
    """
    with open(save_directory+ "vyanjan2_matra.txt",'w',encoding='UTF-8') as f:
        for v1 in vyanjan:
            for v2 in vyanjan:
                for m in matras:
                    f.write(v1 + halanth + v2 + m + "\n")

def vyanjan2_matra2(save_directory):
    """
    Function to generate corpus of 2 vyanjans (1 half, 1 full) with 2 matras
    """
    with open(save_directory+ "vyanjan2_matra2.txt",'w',encoding='UTF-8') as f:
        for v1 in vyanjan:
            for v2 in vyanjan:
                for m1 in matras:
                    for m2 in matras:
                        if m1 != m2 and matra_sanity(m1,m2):
                            f.write(v1 + halanth + v2 + m1 + m2 + "\n")

def vyanajan3_matra(save_directory):
    with open(save_directory+ "vyanja3_matra.txt",'w',encoding='UTF-8') as f:
        for v1 in vyanjan:
            for v2 in vyanjan:
                for v3 in vyanjan:
                    for m1 in matras:
                        f.write(v1 + halanth + v2 + halanth + v3 + m1 + "\n")

def vyanjan3_matra2(save_directory):
    with open(save_directory+ "vyanja3_matra2.txt",'w',encoding='UTF-8') as f:
        for v1 in vyanjan:
            for v2 in vyanjan:
                for v3 in vyanjan:
                    for m1 in matras:
                        for m2 in matras:
                            if m1 != m2 and matra_sanity(m1, m2):
                                f.write(v1 + halanth + v2 + halanth + v3 + m1 + m2 + "\n")

def ank_only(save_directory):
    """
    Function to generate corpus of anks (digits)
    """
    with open(save_directory+ "ank.txt",'w',encoding='UTF-8') as f:
        for n in ank:
            f.write(n + "\n")

def chihn_only(save_directory):
    """
    Function to generate corpus of chihn (punctuation) & special characters
    """
    with open(save_directory+ "chihn.txt",'w',encoding='UTF-8') as f:
        for c in chihn:
            f.write(c + "\n")

def matra(save_directory):
    """
    Function to generate corpus of matras
    """
    with open(save_directory + "matra.txt",'w',encoding='UTF-8') as f:
        for m in matras:
            f.write(m + "\n")

if __name__ == "__main__":
    save_directory = "/home/nrohit/Glyphs/HindiGlyphSynth/Corpus/"
    print(len(vyanjan), len(svar), len(matras), len(chihn))
    vyanjan_matra(save_directory)
    vyanjan_only(save_directory)
    svar_only(save_directory)
    vyanjan_matra2(save_directory)
    svar_matra(save_directory)
    vyanjan2(save_directory)
    vyanjan2_matra(save_directory)
    vyanjan2_matra2(save_directory)
    ank_only(save_directory)
    chihn_only(save_directory)
    # matra(save_directory)
    vyanajan3_matra(save_directory)
    vyanjan3_matra2(save_directory)
