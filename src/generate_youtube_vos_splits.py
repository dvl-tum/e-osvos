import os
import json
import random


seed = 123
random.seed(seed)
data_dir = 'data/YouTube-VOS'


#
# TRAIN
#

num_train_dev_train_val_seqs = 100
num_train_dev_val_seqs = 100


split_dir = os.path.join(data_dir, 'train')

meta_file_path = os.path.join(split_dir, 'meta.json')
with open(meta_file_path, 'r') as f:
    meta_data = json.load(f)['videos']

print(f"Number of sequences in {split_dir}: {len(meta_data)}")

categories_per_seqs = {}

for seq_name, seq_meta_data in meta_data.items():
    for obj_id, obj_meta_data in seq_meta_data['objects'].items():
        obj_category = obj_meta_data['category']

        if not obj_category in categories_per_seqs:
            categories_per_seqs[obj_category] = []

        categories_per_seqs[obj_category].append(seq_name)


sorted_categories = sorted([(c, len(seqs)) for c, seqs in categories_per_seqs.items()],
                           key=lambda x: x[1],
                           reverse=True)

train_dev_val_seqs = []
for i in range(num_train_dev_val_seqs):
    category = sorted_categories[i % len(sorted_categories)][0]
    rnd_seq_idx = random.randint(0, len(categories_per_seqs[category]) - 1)
    rnd_seq = categories_per_seqs[category][rnd_seq_idx]
    train_dev_val_seqs.append(rnd_seq)

    for c, seqs in categories_per_seqs.items():
        if rnd_seq in seqs:
            categories_per_seqs[c] = [s for s in seqs if s != rnd_seq]


train_dev_train_val_seqs = []
for i in range(num_train_dev_train_val_seqs):
    category = sorted_categories[i % len(sorted_categories)][0]
    rnd_seq_idx = random.randint(0, len(categories_per_seqs[category]) - 1)
    rnd_seq = categories_per_seqs[category][rnd_seq_idx]
    train_dev_train_val_seqs.append(rnd_seq)

    for c, seqs in categories_per_seqs.items():
        if rnd_seq in seqs:
            categories_per_seqs[c] = [s for s in seqs if s != rnd_seq]

train_dev_train_seqs = []
for _, seqs in categories_per_seqs.items():
    train_dev_train_seqs.extend(seqs)
train_dev_train_seqs = list(dict.fromkeys(train_dev_train_seqs))


print(len(train_dev_train_seqs))
print(train_dev_val_seqs)
print(train_dev_train_val_seqs)

with open(os.path.join(data_dir, f'train_dev_random_{seed}_train_seqs.txt'), 'w') as f:
    for item in train_dev_train_seqs:
        f.write("%s\n" % item)

with open(os.path.join(data_dir, f'train_dev_random_{seed}_val_seqs.txt'), 'w') as f:
    for item in train_dev_val_seqs:
        f.write("%s\n" % item)

with open(os.path.join(data_dir, f'train_dev_random_{seed}_train_val_seqs.txt'), 'w') as f:
    for item in train_dev_train_val_seqs:
        f.write("%s\n" % item)

#
# VALID
#

# num_splits = 16

# split_file = os.path.join(data_dir, 'valid_seqs.txt')

# with open(split_file) as f:
#     seqs_keys = [seq.strip() for seq in f.readlines()]

# print(f"Number of sequences in {split_file}: {len(seqs_keys)}")

# # print(len(seqs_keys))

# split_length = (len(seqs_keys) // num_splits) + 1

# for i in range(num_splits):
#     split_seqs = seqs_keys[i * split_length: i * split_length + split_length]
#     print(len(split_seqs))
#     with open(os.path.join(data_dir, f'valid_split_{i + 1}_{num_splits}_seqs.txt'), 'w') as the_file:
#         for split_seq in split_seqs:
#             the_file.write(f"{split_seq}\n")
