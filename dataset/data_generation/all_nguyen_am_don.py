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

nguyen_am_don = 'a|ă|â|e|ê|i|y|o|ô|ơ|u|ư'

keyboardNeighbors = getKeyboardNeighbors()
for key in keyboardNeighbors.keys():
    keyboardNeighbors[key] = keyboardNeighbors[key][0][np.argmax(keyboardNeighbors[key][1])]

result = set()
for am_don in nguyen_am_don.split("|"):
    result.add(am_don)
    for candidate in keyboardNeighbors[am_don]:
        result.add(candidate)

print("|".join(result))