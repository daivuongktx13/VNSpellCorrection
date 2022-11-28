import re 
from normalize import chuan_hoa_dau_tu_tieng_viet
import numpy as np
from keyboard_neighbor import getKeyboardNeighbors

with open("common-vietnamese-syllables.txt", "r") as file:
    vi_syllables = [line.strip("\n") for line in file.readlines()]

vi_syllables_new = []
for syllable in vi_syllables:
    normalized = chuan_hoa_dau_tu_tieng_viet(syllable)
    vi_syllables_new.append(normalized)

nguyen_am_ba = 'oai|oao|uao|oeo|iêu|yêu|uôi|ươu|uyu|uyê|ươi|oay|uây|ươi|uya'

keyboardNeighbors = getKeyboardNeighbors()
for key in keyboardNeighbors.keys():
    keyboardNeighbors[key] = keyboardNeighbors[key][0][np.argmax(keyboardNeighbors[key][1])]

result = set()
for am_ba in nguyen_am_ba.split("|"):
    result.add(am_ba)
    if am_ba == "uyê":
        for candidate in keyboardNeighbors[am_ba[2]]:
            result.add(am_ba[0] + am_ba[1] + candidate)
    else:
        for candidate in keyboardNeighbors[am_ba[1]]:
            result.add(am_ba[0] + candidate + am_ba[2])

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


