from random import shuffle

all = open("filelists/all.list").readlines()

shuffle(all)

val = all[-128:]
train = all[:-128]

with open("filelists/val.list", "w") as f:
    for line in val:
        f.write(line)

with open("filelists/train.list", "w") as f:
    for line in train:
        f.write(line)



