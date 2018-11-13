import numpy as np

train_data = np.load("./data/train.npy", encoding='bytes')
train_label = np.load("./data/train_transcripts.npy", encoding='bytes') # 24724 utterances

train_label = train_label

# Convert bytes to string
for i in range(len(train_label)):
    train_label[i] = train_label[i].astype(str)

# character based label
LABEL_LIST = ['0'] # 33 characters
LABEL_MAP = {'0':0}
for i in range(len(train_label)):
    for j in range(len(train_label[i])):
        for char in list(train_label[i][j]):
            if char not in LABEL_LIST:
                LABEL_LIST.append(char)
                LABEL_MAP[char] = len(LABEL_LIST) - 1
LABEL_LIST.append('<space>')
LABEL_MAP['<space>'] = len(LABEL_LIST) - 1
print(LABEL_LIST)
print(LABEL_MAP)
# print(train_label)

char_label = []
for i in range(len(train_label)):
    tmp_label = []
    tmp_label.append(0)
    for j in range(len(train_label[i])):
        for k in range(len(list(train_label[i][j]))):
            key = list(train_label[i][j])[k]
            tmp_label.append(LABEL_MAP[key])
        if j != len(train_label[i]) - 1:
            tmp_label.append(LABEL_MAP['<space>'])
    tmp_label.append(0)
    char_label.append(tmp_label)
# print(char_label)

np.save('./data/train_char.npy', np.array(char_label))

assert len(LABEL_LIST) == len(LABEL_MAP)
assert len(set(LABEL_LIST)) == len(LABEL_MAP)

data = np.load('./data/train_char.npy')
assert len(data) == len(train_data)