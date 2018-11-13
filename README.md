# End-to-End Speech-to-Text System Based on Listen, Attend and Spell

## Description
This is a Pytorch implementation of an attention-based end-to-end speech-to-text system. The deep neural network model is based on Listen, Atten and Spell (LAS). We modified the attention mechanism and slightly change the Listener networks.
## Dataset
We use WSJ (Wall Street Journal) dataset.

Training set: 24724 utterances.

Validation set: 1106 utterances.

Test set: 

## Experiment
### Transcript Preprocessing
1. Convert training transcript from bytes to string.
2. Create a character-based vocabulary from training set.
3. Add \<sos> and \<eos> to the vocabulary. Here we count \<sos> and \<eos> as the same character.
4. Convert words in transcript to characters, and label with corresponding number.
Example:
```python3
utteance = ['THE','FEMALE']
utterance = [<sos>,'T','H','E',<space>,'F','E','M','A','L','E',<eos>]
utterance = [0,y,y,y,y,y,y,y,y,y,y,0]
```
(y is number corresponding to the character)

The reason why we choose character-based model is because it can predict rare words.