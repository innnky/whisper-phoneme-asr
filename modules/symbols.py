
phones = set()
pinyins = set()
c_set = set()
v_set = set()
for line in open("assets/opencpop-strict.txt").readlines():
    pinyin, phs =line.strip().split("\t")
    pinyins.add(pinyin)
    phs = phs.split(" ")
    if len(phs) == 1:
        v_set.add(phs[0])
    else:
        c_set.add(phs[0])
        [v_set.add(i) for i in phs[1:]]
    [phones.add(i) for i in phs]


phone_set = ['_'] + sorted(phones) + ["SP"]

# ['_', 'E', 'En', 'a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h',
# 'i', 'i0', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'ir', 'iu', 'j', 'k', 'l', 'm', 'n', 'o', 'ong',
# 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn',
# 'w', 'x', 'y', 'z', 'zh', 'SP']

phone_to_int = {}
int_to_phone = {}
for idx, item in enumerate(phone_set):
    phone_to_int[item] = idx
    int_to_phone[idx] = item


