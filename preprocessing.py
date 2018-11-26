import numpy as np

########## Preprocessing Training Dataset ##########
train_data = np.load("./data/train.npy", encoding='bytes')
train_label = np.load("./data/train_transcripts.npy", encoding='bytes') # 24724 utterances
print("train label", len(train_label)) # 1106 utterances

# Convert bytes to string
for i in range(len(train_label)):
    train_label[i] = train_label[i].astype(str)

# Create vocabulary list and vocabulary map
# character-based label
LABEL_LIST = ['<sos>', '<eos>'] # 34 characters
LABEL_MAP = {'<eos>': 0, '<sos>': 1}
for i in range(len(train_label)):
    for j in range(len(train_label[i])):
        for char in list(train_label[i][j]):
            if char not in LABEL_LIST:
                LABEL_LIST.append(char)
                LABEL_MAP[char] = len(LABEL_LIST) - 1
LABEL_LIST.append('<space>')
LABEL_MAP['<space>'] = len(LABEL_LIST) - 1
LABEL_LIST = sorted(LABEL_LIST)
for key in LABEL_MAP.keys():
    LABEL_MAP[key] = LABEL_LIST.index(key)

print(LABEL_LIST)
print(LABEL_MAP)

# Convert transcripts to character-based label
char_label = []
for i in range(len(train_label)):
    tmp_label = []
    tmp_label.append(LABEL_MAP['<sos>'])
    for j in range(len(train_label[i])):
        for k in range(len(list(train_label[i][j]))):
            key = list(train_label[i][j])[k]
            tmp_label.append(LABEL_MAP[key])
        if j != len(train_label[i]) - 1:
            tmp_label.append(LABEL_MAP['<space>'])
    tmp_label.append(LABEL_MAP['<eos>'])
    char_label.append(tmp_label)

# Save characters-based label to file
np.save('./data/train_char.npy', np.array(char_label))

assert len(LABEL_LIST) == len(LABEL_MAP)
assert len(set(LABEL_LIST)) == len(LABEL_MAP)

data = np.load('./data/train_char.npy')
assert len(data) == len(train_data)

########## Preprocessing Validation Dataset ##########
dev_label = np.load("./data/dev_transcripts.npy", encoding='bytes')
print("dev label", len(dev_label)) # 1106 utterances

# Convert bytes to string
for i in range(len(dev_label)):
    dev_label[i] = dev_label[i].astype(str)

# Convert transcripts to character-based label
dev_char_label = []
for i in range(len(dev_label)):
    tmp_label = []
    tmp_label.append(LABEL_MAP['<sos>'])
    for j in range(len(dev_label[i])):
        for k in range(len(list(dev_label[i][j]))):
            key = list(dev_label[i][j])[k]
            tmp_label.append(LABEL_MAP[key])
        if j != len(dev_label[i]) - 1:
            tmp_label.append(LABEL_MAP['<space>'])
    tmp_label.append(LABEL_MAP['<eos>'])
    dev_char_label.append(tmp_label)

# Save characters-based label to file
np.save('./data/dev_char.npy', np.array(dev_char_label))