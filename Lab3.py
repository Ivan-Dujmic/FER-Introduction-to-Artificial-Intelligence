import sys
import math
from collections import defaultdict # So we don't have to check if the key exists

# Terminology reminder:
# Feature - attribute of the data
# Example - a single row of data (a single tuple of feature values)
# Class - what we are classifying / predicting
# Label - the value of the class

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        feature_names = header[:-1]
        class_name = header[-1]

        data = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            row = {feature_names[i]: parts[i] for i in range(len(feature_names))}
            row[class_name] = parts[-1]
            data.append(row)
        return feature_names, class_name, data

class ID3_decision_tree:
    def __init__(self, max_depth=None):
        self.tree = None
        self.data = None
        self.class_name = None
        self.max_depth = max_depth

    # Returns the most common label of a class in the data 
    def _most_common_label(self, data):
        label_counts = defaultdict(int)
        for row in data:
            label_counts[row[self.class_name]] += 1
        # Max of the counts, if two labels have the same count, return the one that comes first alphabetically
        # Using min and negative count because there's no simple way to reverse alphabetical order
        return min(label_counts.items(), key=lambda x: (-x[1], x[0].lower()))[0]

    def _print_path(self, path, label):
        index = 1
        for feature, value in path:
            print(f"{index}:{feature}={value}", end=" ")
            index += 1
        print(f"{label}")

    def _id3(self, data, parent_data, feature_names, path, depth):
        # If no data left, return the most common label in the parent data
        if len(data) == 0:
            label = self._most_common_label(parent_data)
            self._print_path(path, label)
            return label
        
        # If no features left or all examples have the same label (entropy is zero)
        label = self._most_common_label(data)
        if len(feature_names) == 0 or all(row[self.class_name] == label for row in data):
            self._print_path(path, label)
            return label
        
        # If max depth is reached, return the most common label
        if self.max_depth is not None and depth >= self.max_depth:
            label = self._most_common_label(data)
            self._print_path(path, label)
            return label
        
        # Pick the best feature to split on (highest information gain <-- lowest new entropy)
        lowest_new_entropy = float('inf')
        best_feature = None
        for feature in feature_names:
            # Splits the data by values of a feature
            partitions = defaultdict(list)
            for row in data:
                partitions[row[feature]].append(row)
            new_entropy = 0
            for partition in partitions.values():
                # Part of data that this partition represents and it's entropy
                prob = len(partition) / len(data)
                if prob == 0:
                    continue
                else:
                    # Count the labels in the partition
                    label_counts = defaultdict(int)
                    for row in partition:
                        label_counts[row[self.class_name]] += 1
                    entropy = 0
                    for count in label_counts.values():
                        prob_label = count / len(partition)
                        entropy -= prob_label * math.log2(prob_label) if prob_label > 0 else 0
                    new_entropy += prob * entropy
            if new_entropy < lowest_new_entropy:
                lowest_new_entropy = new_entropy
                best_feature = feature
            elif new_entropy == lowest_new_entropy: # If two features have the same entropy, pick the one that comes first alphabetically
                if feature < best_feature:
                    best_feature = feature
        
        subtrees = defaultdict(list)

        # Call id3 recursively for each value of the best feature
        feature_values = set(row[best_feature] for row in data)
        for value in feature_values:
            # Get the subset of data that has this value for the best feature
            data_subset = [row for row in data if row[best_feature] == value]
            # Remove the best feature from the list of features
            new_feature_names = [f for f in feature_names if f != best_feature]
            # Recursively call id3 on the subset
            subtree = self._id3(data_subset, data, new_feature_names, path + [(best_feature, value)], depth + 1)
            subtrees[value] = subtree
        return {best_feature: subtrees}

    def fit(self, data, feature_names, class_name):
        self.data = data
        self.class_name = class_name
        print("[BRANCHES]:")
        self.tree = self._id3(data, data, feature_names, [], 0)

    def predict(self, data):
        print("[PREDICTIONS]:", end=" ")
        correct = 0
        confusion = defaultdict(lambda: defaultdict(int))
        for row in data:
            path = []
            current_node = self.tree
            while isinstance(current_node, dict):
                feature = list(current_node.keys())[0]
                value = row[feature]
                # If the tree doesn't have a decision for the value then find the most common label in the parent data
                if value not in current_node[feature]:
                    parent_data = data
                    for node in path:
                        parent_data = [r for r in parent_data if r[node[0]] == node[1]]
                    label = self._most_common_label(parent_data)
                    print(f"{label}", end=" ")
                    if label == row[self.class_name]:
                        correct += 1
                    confusion[row[self.class_name]][label] += 1
                    break
                current_node = current_node[feature][value]
                path.append((feature, value))
            else:   # If we didn't break out of the loop, we have a leaf node
                print(f"{current_node}", end=" ")
                if current_node == row[self.class_name]:
                    correct += 1
                confusion[row[self.class_name]][current_node] += 1
        print(f"\n[ACCURACY]: {correct / len(data):.5f}")
        print("[CONFUSION_MATRIX]:")
        for true_label in sorted(confusion.keys()):
            for predicted_label in sorted(confusion.keys()):    # We're using the same labels for both axes (otherwise we'd be missing the ones we never predicted)
                print(f"{confusion[true_label][predicted_label]}", end=" ")
            print("", end="\n")

def main():
    training_data_file_path = sys.argv[1]
    test_data_file_path = sys.argv[2]
    depth = None
    if len(sys.argv) > 3:
        depth = int(sys.argv[3])    # Optional

    feature_names, class_name, training_data = read_data(training_data_file_path)
    test_feature_names, test_class_name, test_data = read_data(test_data_file_path)

    model = ID3_decision_tree(depth)
    model.fit(training_data, feature_names, class_name)
    model.predict(test_data)

if __name__ == "__main__":
    main()