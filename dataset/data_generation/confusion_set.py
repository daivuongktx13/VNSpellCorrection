import re 
from normalize import chuan_hoa_dau_tu_tieng_viet
import numpy as np
from tqdm import tqdm
import textdistance
import json
from copy import copy
with open("common-vietnamese-syllables.txt", "r") as file:
    vi_syllables = [line.strip("\n") for line in file.readlines()]

vi_syllables_new = []
for syllable in vi_syllables:
    normalized = chuan_hoa_dau_tu_tieng_viet(syllable)
    vi_syllables_new.append(normalized)

regex_nguyen_am_don = "ộ|ặ|ằ|ụ|ầ|a|ũ|á|ể|ỡ|ủ|y|ở|ế|ẵ|ệ|é|ẹ|â|ề|ê|ọ|ờ|ẳ|ợ|ỷ|ữ|ị|e|u|ò|ẫ|i|ỉ|ẩ|ẽ|õ|ỹ|ô|ỵ|ồ|ú|í|ó|ỗ|ã|ẻ|ù|ă|ơ|ứ|ậ|ử|ừ|à|ĩ|ả|ố|ớ|ự|ắ|o|ý|ỳ|ư|ấ|ễ|ạ|ỏ|ổ|è|ì"
regex_nguyen_am_doi = "uằ|iê|ấu|ượ|ùy|ạy|uỹ|ươ|ỗi|yệ|ụy|ẫy|oà|ái|ói|uồ|uỷ|oỏ|ệu|ue|oi|ậu|oè|uã|ãi|òi|ơi|ựa|ụi|iể|oá|ìa|ĩu|uẹ|ìu|ầu|ỏe|ối|uẳ|ịa|òe|ai|ọe|yể|ày|ỉu|uỵ|uể|óe|ỉa|ũa|ườ|uè|êu|ẹo|uá|ỏi|uấ|ưỡ|ội|au|iề|ửu|ọi|ảu|uẽ|ầy|ẻo|ao|yế|uẻ|uơ|ưở|iế|uở|ịu|ủa|ẫu|uặ|oằ|oò|ạu|uỳ|ạo|oọ|ưa|oẹ|ui|uậ|ủi|áo|óa|ẩu|ảy|oẵ|áu|ựu|uô|ửa|ễu|uâ|oạ|uổ|uê|ùi|ếu|ời|iu|uo|oé|yễ|oẳ|uớ|ay|iễ|ủy|ướ|oó|eo|ũi|oả|ua|ỏa|ấy|uố|èo|oo|úy|ẩy|ồi|yề|ẽo|uẫ|ứu|ãy|ổi|ía|ảo|ué|uờ|ùa|ia|ều|oa|iệ|àu|õa|oắ|uắ|uả|ứa|ởi|ụa|ũy|òa|íu|éo|oã|uă|uộ|ữu|úa|ải|ỡi|ừu|ểu|oe|õi|ọa|ừa|uệ|uý|uó|ào|uà|ây|oă|uạ|ữa|oặ|uy|ợi|uẩ|uỗ|ão|uế|ưu|ửi|ại|âu|ới|uầ|ĩa|úi|oẻ|ôi|ài|uề|yê|ậy|áy"
regex_nguyen_am_ba = "uỷu|uây|ươu|iệu|yếu|yểu|uyế|uyệ|uyề|ưỡi|uôi|ượi|uổi|oay|uào|iễu|oeo|oèo|uỗi|oai|uấy|oái|uỵu|uyể|uồi|oáy|yều|oẹo|uẫy|ưởi|iểu|uầy|iêu|uối|uyễ|ưới|iều|oài|uao|ươi|yêu|ười|uya|oải|ướu|uội|oại|iếu|ượu|uẩy|uyê|uậy"
all_phu_am_dau = {'', 'gh', 'q', 'kh', 'p', 'm', 'qu', 'n', 'b', 'g', 't', 'ch', 'th', 'k', 'đ', 'r', 'ph', 'ngh', 'gi', 'tr', 's', 'l', 'h', 'nh', 'c', 'ng', 'd', 'v', 'x'}
all_phu_am_cuoi = {'', 'ng', 'nh', 't', 'ch', 'c', 'p', 'm', 'k', 'n'}
all_nguyen_am_don = "ộ|ặ|ằ|ụ|ầ|a|ũ|á|ể|ỡ|ủ|y|ở|ế|ẵ|ệ|é|ẹ|â|ề|ê|ọ|ờ|ẳ|ợ|ỷ|ữ|ị|e|u|ò|ẫ|i|ỉ|ẩ|ẽ|õ|ỹ|ô|ỵ|ồ|ú|í|ó|ỗ|ã|ẻ|ù|ă|ơ|ứ|ậ|ử|ừ|à|ĩ|ả|ố|ớ|ự|ắ|o|ý|ỳ|ư|ấ|ễ|ạ|ỏ|ổ|è|ì".split("|")
all_nguyen_am_doi = "uằ|iê|ấu|ượ|ùy|ạy|uỹ|ươ|ỗi|yệ|ụy|ẫy|oà|ái|ói|uồ|uỷ|oỏ|ệu|ue|oi|ậu|oè|uã|ãi|òi|ơi|ựa|ụi|iể|oá|ìa|ĩu|uẹ|ìu|ầu|ỏe|ối|uẳ|ịa|òe|ai|ọe|yể|ày|ỉu|uỵ|uể|óe|ỉa|ũa|ườ|uè|êu|ẹo|uá|ỏi|uấ|ưỡ|ội|au|iề|ửu|ọi|ảu|uẽ|ầy|ẻo|ao|yế|uẻ|uơ|ưở|iế|uở|ịu|ủa|ẫu|uặ|oằ|oò|ạu|uỳ|ạo|oọ|ưa|oẹ|ui|uậ|ủi|áo|óa|ẩu|ảy|oẵ|áu|ựu|uô|ửa|ễu|uâ|oạ|uổ|uê|ùi|ếu|ời|iu|uo|oé|yễ|oẳ|uớ|ay|iễ|ủy|ướ|oó|eo|ũi|oả|ua|ỏa|ấy|uố|èo|oo|úy|ẩy|ồi|yề|ẽo|uẫ|ứu|ãy|ổi|ía|ảo|ué|uờ|ùa|ia|ều|oa|iệ|àu|õa|oắ|uắ|uả|ứa|ởi|ụa|ũy|òa|íu|éo|oã|uă|uộ|ữu|úa|ải|ỡi|ừu|ểu|oe|õi|ọa|ừa|uệ|uý|uó|ào|uà|ây|oă|uạ|ữa|oặ|uy|ợi|uẩ|uỗ|ão|uế|ưu|ửi|ại|âu|ới|uầ|ĩa|úi|oẻ|ôi|ài|uề|yê|ậy|áy".split("|")
all_nguyen_am_ba = "uỷu|uây|ươu|iệu|yếu|yểu|uyế|uyệ|uyề|ưỡi|uôi|ượi|uổi|oay|uào|iễu|oeo|oèo|uỗi|oai|uấy|oái|uỵu|uyể|uồi|oáy|yều|oẹo|uẫy|ưởi|iểu|uầy|iêu|uối|uyễ|ưới|iều|oài|uao|ươi|yêu|ười|uya|oải|ướu|uội|oại|iếu|ượu|uẩy|uyê|uậy".split("|")

confusion_set = dict()

special_list = set()
for syllable in tqdm(vi_syllables_new):
    # print(syllable)
    if syllable[0:2] in ["qu", "gi"]:
        special_list.add(syllable)
        # print(f"Ignore {syllable}")
        continue

    confusion_set[syllable] = dict()
    syllable_candidates = confusion_set[syllable]
    syllable_candidates['phu_am_dau'] = set()
    syllable_candidates['nguyen_am'] = set()
    syllable_candidates['phu_am_cuoi'] = set()

    if len(re.findall(regex_nguyen_am_ba, syllable)) != 0:
        result = re.findall(regex_nguyen_am_ba, syllable)
        nguyen_am = result[0]
    elif len(re.findall(regex_nguyen_am_doi, syllable)) != 0:
        result = re.findall(regex_nguyen_am_doi, syllable)
        nguyen_am = result[0]
    elif len(re.findall(regex_nguyen_am_don, syllable)) != 0:
        result = re.findall(regex_nguyen_am_don, syllable)
        nguyen_am = result[0]
    else:
        raise Exception("Khong co nguyen am")
    phu_am_dau, phu_am_cuoi = "", ""
    if len(re.findall(f"(.+){nguyen_am}", syllable)) !=0 :
        result = re.findall(f"(.+){nguyen_am}", syllable)
        phu_am_dau = result[0]
    if len(re.findall(f"{nguyen_am}(.+)", syllable)) !=0 :
        result = re.findall(f"{nguyen_am}(.+)", syllable)
        phu_am_cuoi = result[0]

    ### Error thay đổi phụ âm đầu
    for candidate in all_phu_am_dau:
        if "".join([candidate, nguyen_am, phu_am_cuoi]) in vi_syllables_new:
            syllable_candidates['phu_am_dau'].add("".join([candidate, nguyen_am, phu_am_cuoi]))
    ### Error thay đổi nguyên âm
    all_nguyen_am = all_nguyen_am_don + all_nguyen_am_doi + all_nguyen_am_ba
    for candidate in all_nguyen_am:
        if "".join([phu_am_dau, candidate, phu_am_cuoi]) in vi_syllables_new:
            syllable_candidates['nguyen_am'].add("".join([phu_am_dau, candidate, phu_am_cuoi]))
    ### Error thay đổi phụ âm cuối
    for candidate in all_phu_am_cuoi:
        if "".join([phu_am_dau, nguyen_am, candidate]) in vi_syllables_new:
            syllable_candidates['phu_am_cuoi'].add("".join([phu_am_dau, nguyen_am, candidate]))

for syllable in tqdm(special_list):

    if len(re.findall(regex_nguyen_am_don, syllable)) > 1:
        phu_am_dau = syllable[0:2]
        remained = syllable[2:]
    else:
        phu_am_dau = syllable[0]
        remained = syllable[1:]

    confusion_set[syllable] = dict()
    syllable_candidates = confusion_set[syllable]
    syllable_candidates['phu_am_dau'] = set()
    syllable_candidates['nguyen_am'] = set()
    syllable_candidates['phu_am_cuoi'] = set()
    

    if len(re.findall(regex_nguyen_am_ba, remained)) != 0:
        result = re.findall(regex_nguyen_am_ba, remained)
        nguyen_am = result[0]
    elif len(re.findall(regex_nguyen_am_doi, remained)) != 0:
        result = re.findall(regex_nguyen_am_doi, remained)
        nguyen_am = result[0]
    elif len(re.findall(regex_nguyen_am_don, remained)) != 0:
        result = re.findall(regex_nguyen_am_don, remained)
        nguyen_am = result[0]
    else:
        nguyen_am, phu_am_cuoi = "", ""
        
    phu_am_cuoi = ""

    if nguyen_am != "" and len(re.findall(f"{nguyen_am}(.+)", remained)) !=0 :
        result = re.findall(f"{nguyen_am}(.+)", remained)
        phu_am_cuoi = result[0]

    ### Error thay đổi phụ âm đầu
    for candidate in all_phu_am_dau:
        if "".join([candidate, nguyen_am, phu_am_cuoi]) in vi_syllables_new:
            syllable_candidates['phu_am_dau'].add("".join([candidate, nguyen_am, phu_am_cuoi]))
    ### Error thay đổi nguyên âm
    all_nguyen_am = all_nguyen_am_don + all_nguyen_am_doi + all_nguyen_am_ba
    for candidate in all_nguyen_am:
        if "".join([phu_am_dau, candidate, phu_am_cuoi]) in vi_syllables_new:
            syllable_candidates['nguyen_am'].add("".join([phu_am_dau, candidate, phu_am_cuoi]))
    ### Error thay đổi phụ âm cuối
    for candidate in all_phu_am_cuoi:
        if "".join([phu_am_dau, nguyen_am, candidate]) in vi_syllables_new:
            syllable_candidates['phu_am_cuoi'].add("".join([phu_am_dau, nguyen_am, candidate]))

for key in tqdm(confusion_set.keys()):
    for key_2_level in confusion_set[key].keys():
        try:
            confusion_set[key][key_2_level].remove(key)
        except:
            pass

for key in tqdm(confusion_set.keys()):
    for key_2_level in confusion_set[key].keys():
        candidates_to_remove = []
        for candidate in confusion_set[key][key_2_level]:
            similarity = textdistance.damerau_levenshtein.normalized_similarity(key, candidate)
            if similarity < 0.5:
                candidates_to_remove.append(candidate)
        for candidate in candidates_to_remove:
            confusion_set[key][key_2_level].remove(candidate)

keyboard_neighbor = {'a': 'áàảãạ',
 'ă': 'ắằẳẵặ',
 'â': 'ấầẩẫậ',
 'á': 'aàảãạ',
 'à': 'aáảãạ',
 'ả': 'aáàãạ',
 'ã': 'aáàảạ',
 'ạ': 'aáàảã',
 'ắ': 'ăằẳẵặ',
 'ằ': 'ăắẳẵặ',
 'ẳ': 'ăắằẵặ',
 'ặ': 'ăắằẳẵ',
 'ẵ': 'ăắằẳặ',
 'ấ': 'âầẩẫậ',
 'ầ': 'âấẩẫậ',
 'ẩ': 'âấầẫậ',
 'ẫ': 'âấầẩậ',
 'ậ': 'âấầẩẫ',
 'e': 'èéẻẽẹ',
 'é': 'eèẻẽẹ',
 'è': 'eéẻẽẹ',
 'ẻ': 'eéèẽẹ',
 'ẽ': 'eéèẻẹ',
 'ẹ': 'eéèẻẽ',
 'ê': 'ếềểễệ',
 'ế': 'êềểễệ',
 'ề': 'êếểễệ',
 'ể': 'êếềễệ',
 'ễ': 'êếềểệ',
 'ệ': 'êếềểễ',
 'i': 'íìỉĩị',
 'í': 'iìỉĩị',
 'ì': 'iíỉĩị',
 'ỉ': 'iíìĩị',
 'ĩ': 'iíìỉị',
 'ị': 'iíìỉĩ',
 'o': 'òóỏọõ',
 'ó': 'oòỏọõ',
 'ò': 'oóỏọõ',
 'ỏ': 'oóòọõ',
 'õ': 'oóòỏọ',
 'ọ': 'oóòỏõ',
 'ô': 'ốồổỗộ',
 'ố': 'ôồổỗộ',
 'ồ': 'ôốổỗộ',
 'ổ': 'ôốồỗộ',
 'ộ': 'ôốồổỗ',
 'ỗ': 'ôốồổộ',
 'ơ': 'ớờởợỡ',
 'ớ': 'ơờởợỡ',
 'ờ': 'ơớởợỡ',
 'ở': 'ơớờợỡ',
 'ợ': 'ơớờởỡ',
 'ỡ': 'ơớờởợ',
 'u': 'úùủũụ',
 'ú': 'uùủũụ',
 'ù': 'uúủũụ',
 'ủ': 'uúùũụ',
 'ũ': 'uúùủụ',
 'ụ': 'uúùủũ',
 'ư': 'ứừữửự',
 'ứ': 'ưừữửự',
 'ừ': 'ưứữửự',
 'ử': 'ưứừữự',
 'ữ': 'ưứừửự',
 'ự': 'ưứừữử',
 'y': 'ýỳỷỵỹ',
 'ý': 'yỳỷỵỹ',
 'ỳ': 'yýỷỵỹ',
 'ỷ': 'yýỳỵỹ',
 'ỵ': 'yýỳỷỹ',
 'ỹ': 'yýỳỷỵ'}

pattern = "(" + "|".join(keyboard_neighbor.keys()) + "){1}"

def make_accent_change_candidates(text):
    result = re.findall(pattern, text)
    candidates =  []
    for candidate in result:
        [candidates.append(text.replace(candidate, x)) for x in keyboard_neighbor[candidate]]
    return set(candidates)

typo = json.load(open("../noising_resources/typo.json", "r"))
typo_pattern = "(" + "|".join(typo.keys()) + "){1}"
accent_pattern = "(s|f|r|x|j|1|2|3|4|5){1}"

def convert_to_non_telex(text):
    word = copy(text)
    candidates = re.findall(typo_pattern, text)
    for candidate in candidates:
        replaced = typo[candidate][0]
            # Move accent to the end of text
        if len(re.findall(accent_pattern, replaced)) != 0:
            word = re.sub(candidate, replaced[0:-1], word)
            word += replaced[-1]
        else:
            word = re.sub(candidate, replaced, word)
    return word


def keep_1_distance_candidates(text, nguyen_am_errors : set):
    nguyen_am_errors = list(nguyen_am_errors)
    text = convert_to_non_telex(text)
    distances = [textdistance.damerau_levenshtein(text, convert_to_non_telex(error)) for error in nguyen_am_errors]
    indies_to_keep = np.where(np.array(distances) <= 1)[0]
    return set([nguyen_am_errors[i] for i in indies_to_keep])

for key in tqdm(confusion_set.keys()):
    candidates = make_accent_change_candidates(key)
    one_distance_candidates = keep_1_distance_candidates(key, confusion_set[key]['nguyen_am'])
    candidates = candidates.union(one_distance_candidates)
    high_probs_list = candidates.intersection(confusion_set[key]['nguyen_am'])
    lower_probs_list = confusion_set[key]['nguyen_am'].difference(high_probs_list)
    confusion_set[key]['nguyen_am'] = [high_probs_list, lower_probs_list]

with open("../noising_resources/confusion_set.json", "w+") as outfile:
    print(confusion_set, file = outfile)

