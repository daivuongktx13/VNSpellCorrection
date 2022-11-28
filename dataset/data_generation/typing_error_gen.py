import re 
from normalize import chuan_hoa_dau_tu_tieng_viet
import numpy as np

with open("common-vietnamese-syllables.txt", "r") as file:
    vi_syllables = [line.strip("\n") for line in file.readlines()]

file = open("../../dataset/noising_resources/kieu_go_dau_cu_moi.txt", "w+")
for syllable in vi_syllables:
    normalized = chuan_hoa_dau_tu_tieng_viet(syllable)
    if normalized != syllable:
        print(normalized, syllable, file = file)
file.close()