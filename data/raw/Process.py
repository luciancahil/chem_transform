input_file = "Train.csv"

char_count = dict()

with open(input_file, mode='r', encoding='utf-8') as file:
    for line in file:
        for c in line:
            if c in char_count:
                char_count[c] = char_count[c] + 1
            else:
                char_count[c] = 1

print(char_count)
print("Num unique chars: " + str(len(char_count)))

char_to_idx = dict()
idx_to_char = []
idx_to_char.append('[START]')

for key in char_count.keys():
    if char_count[key] < 5000:
        continue
    char_to_idx[key] = len(char_to_idx) + 1
    idx_to_char.append(key)

print(char_to_idx)
print(idx_to_char)