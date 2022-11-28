import re 
from normalize import chuan_hoa_dau_tu_tieng_viet
from keyboard_neighbor import getKeyboardNeighbors
import numpy as np

with open("common-vietnamese-syllables.txt", "r") as file:
    vi_syllables = [line.strip("\n") for line in file.readlines()]

vi_syllables_new = []
for syllable in vi_syllables:
    normalized = chuan_hoa_dau_tu_tieng_viet(syllable)
    vi_syllables_new.append(normalized)

nguyen_am_doi = 'ai|ao|au|ay|âu|ây|êu|eo|ia|iê|yê|iu|oă|oa|oi|oe|oo|ôô|ơi|uă|uâ|ue|ua|ui|ưi|uo|ươ|ưu|uơ|uy|ưa|ôi|uô|uê'

no_end_phu_am = 'ưu|ưi|ui|ưa|ơi|ôi|oi|iu|ia|êu|eo|ây|ay|âu|au|ao|ai'
must_end_phu_am = "yê|ươ|uô|uâ|iê|â"

keyboardNeighbors = getKeyboardNeighbors()
for key in keyboardNeighbors.keys():
    keyboardNeighbors[key] = keyboardNeighbors[key][0][np.argmax(keyboardNeighbors[key][1])]


result = set()
for am_doi in nguyen_am_doi.split("|"):
    result.add(am_doi)
    if am_doi not in must_end_phu_am:
        for candidate in keyboardNeighbors[am_doi[0]]:
            result.add(candidate + am_doi[1])
    if am_doi not in no_end_phu_am:
        for candidate in keyboardNeighbors[am_doi[1]]:
                result.add(am_doi[0] + candidate)
        

remove_list = set()
for syllable in result:
    for idx in range(len(vi_syllables_new)):
        if syllable in vi_syllables_new[idx]:
            break
    
    if idx == len(vi_syllables_new) - 1:
        remove_list.add(syllable)

for el in remove_list:
    result.discard(el)

print("|".join(result))
