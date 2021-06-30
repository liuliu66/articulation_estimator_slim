import os

sample_list = []
with open('train.txt', 'r') as f:
    for line in f.readlines():
        sample_list.append(line.split('6/')[-1])

with open('train2.txt', 'w') as f:
    for sample in sample_list:
        f.write(sample)