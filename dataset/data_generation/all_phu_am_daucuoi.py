import re 
from normalize import chuan_hoa_dau_tu_tieng_viet
import numpy as np

with open("common-vietnamese-syllables.txt", "r") as file:
    vi_syllables = [line.strip("\n") for line in file.readlines()]


vi_syllables_new = []
for syllable in vi_syllables:
    normalized = chuan_hoa_dau_tu_tieng_viet(syllable)
    vi_syllables_new.append(normalized)

regex_am_ba = "ười|oeo|uyễ|uồi|ươi|uyê|uôi|ướu|oải|uyệ|oẹo|ưới|iễu|uối|yếu|oại|ưỡi|iêu|ưởi|oèo|uya|oáy|uổi|uỷu|uyế|uyể|ượu|uội|uao|uầy|uào|uẫy|ươu|yểu|oai|uyề|oài|uậy|iều|uỵu|iếu|oay|yều|uấy|oái|iểu|uẩy|yêu|uỗi|iệu|uây|ượi"
regex_am_hai = "áo|ay|ùy|ại|ậu|ỡi|èo|ọi|ào|ao|uấ|ãy|uề|uy|ảu|oạ|iê|ái|ảy|ội|ựa|ẻo|ời|ôi|iệ|oỏ|ủi|ía|oẻ|uệ|ọe|ẫy|ơi|ồi|uẹ|ũy|ấy|ủa|ùa|ỗi|ượ|uý|eo|ấu|ễu|iề|ướ|ưu|ụi|ụy|iễ|uỗ|âu|uồ|ửi|uã|ạo|ây|ia|ìa|àu|ểu|uả|oả|oo|ếu|ĩa|ué|ẽo|oà|uộ|ue|oẹ|uâ|ịu|uố|íu|yể|òe|uằ|uẳ|ùi|au|uo|iu|ựu|iể|uẽ|uở|õi|éo|ão|ới|uậ|uỹ|ìu|yệ|oặ|ui|ầy|yế|áu|óa|yê|ợi|oe|oè|ẫu|uơ|oó|uá|ửu|úa|uầ|ưở|ỏe|ĩu|oé|uể|ậy|úi|ỏi|uà|ủy|oằ|ữa|oã|ửa|uớ|oă|ổi|oò|uă|uắ|uờ|ườ|úy|ữu|ối|uó|oi|ừu|oá|ởi|ừa|ũi|ải|yề|ỉa|uặ|ưa|òa|òi|ệu|ạy|uổ|ịa|uê|ạu|ụa|ãi|oọ|ài|oẳ|uỷ|ưỡ|ẩy|uỳ|iế|ọa|uế|ua|ũa|óe|uẩ|oắ|ẩu|uẻ|ai|ỉu|ói|ầu|ươ|uè|ều|ảo|yễ|êu|uẫ|oa|ứu|ày|uỵ|oẵ|áy|ứa|ỏa|uô|õa|uạ|ẹo"
regex_am_don = "ề|e|a|ầ|è|ơ|ồ|ú|ỵ|ả|ắ|ỷ|ố|ẩ|ặ|ừ|ữ|ủ|ụ|é|ợ|ằ|á|ỉ|ỗ|ê|ờ|ạ|õ|o|y|ì|ỳ|ự|ấ|ế|ý|ẽ|ó|u|ể|ễ|i|â|ẻ|ẹ|ỏ|ớ|ẳ|ẵ|ỹ|à|ẫ|ị|ù|ư|ứ|ở|ộ|ỡ|ũ|ô|í|ổ|ệ|ò|ĩ|ọ|ã|ậ|ử|ă"

all_phu_am_dau = set()
all_phu_am_cuoi = set()
special_list = set()
for syllable in vi_syllables_new:
    if syllable[0:2] in ["qu", "gi"]:
        special_list.add(syllable)
        continue

    if len(result:=re.findall(regex_am_ba, syllable)) != 0:
        nguyen_am = result[0]
    elif len(result:=re.findall(regex_am_hai, syllable)) != 0:
        nguyen_am = result[0]
    elif len(result:=re.findall(regex_am_don, syllable)) != 0:
        nguyen_am = result[0]
    else:
        raise Exception("Khong co nguyen am")
    phu_am_dau, phu_am_cuoi = "", ""
    if len(result:=re.findall(f"(.+){nguyen_am}", syllable)) !=0 :
        phu_am_dau = result[0]
    if len(result:=re.findall(f"{nguyen_am}(.+)", syllable)) !=0 :
        phu_am_cuoi = result[0]

    all_phu_am_dau.add(phu_am_dau)
    all_phu_am_cuoi.add(phu_am_cuoi)
    
    assert "".join([phu_am_dau, nguyen_am, phu_am_cuoi]) == syllable


for syllable in special_list:

    if len(result:=re.findall(regex_am_don, syllable)) > 1:
        phu_am_dau = syllable[0:2]
        remained = syllable[2:]
    else:
        phu_am_dau = syllable[0]
        remained = syllable[1:]
    

    if len(result:=re.findall(regex_am_ba, remained)) != 0:
        nguyen_am = result[0]
    elif len(result:=re.findall(regex_am_hai, remained)) != 0:
        nguyen_am = result[0]
    elif len(result:=re.findall(regex_am_don, remained)) != 0:
        nguyen_am = result[0]
    else:
        nguyen_am, phu_am_cuoi = "", ""
        
    phu_am_cuoi = ""

    if nguyen_am != "" and len(result:=re.findall(f"{nguyen_am}(.+)", remained)) !=0 :
        phu_am_cuoi = result[0]

    all_phu_am_dau.add(phu_am_dau)
    all_phu_am_cuoi.add(phu_am_cuoi)
    
    assert "".join([phu_am_dau, nguyen_am, phu_am_cuoi]) == syllable

print("Tất cả phụ âm đầu: ")
print(all_phu_am_dau)
print("Tất cả phụ âm cuối: ")
print(all_phu_am_cuoi)