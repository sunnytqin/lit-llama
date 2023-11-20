import pickle

with open("val_data.pickle", "rb") as fp:
    data = pickle.load(fp)

small_entropy, ground_truth = data

sorted_pairs = list(sorted(zip(small_entropy, ground_truth), key=lambda x: x[0]))

ones_count = 0
ones_so_far = []
for s, gt in sorted_pairs:
    if(gt == 1):
        ones_count += 1
    
    ones_so_far.append(ones_count)

total_ones = ones_so_far[-1]
best_acc = -1
best_index = -1
for i in range(len(sorted_pairs)):
    zeros_correct = i - ones_so_far[i] + 1
    ones_correct = total_ones - ones_so_far[i]
    acc = (zeros_correct + ones_correct) / len(sorted_pairs)
    if(acc > best_acc):
        best_acc = acc
        best_index = i

print(f"Best acc: {best_acc}")