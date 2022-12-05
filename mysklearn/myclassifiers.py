import operator
import math
import copy
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        row_indexes_dists = []
        for i, train_instance in enumerate(self.X_train):
            dist = myutils.compute_euclidean_distance(train_instance, X_test)
            row_indexes_dists.append((i, dist))

        # now we need the k smallest distances
        # we can sort row_indexes_dists by distance
        row_indexes_dists.sort(key=operator.itemgetter(-1)) # -1 or 1
        # because the distance is at the index in each item (list)
    
        # now, grab the top k
        top_k = row_indexes_dists[:self.n_neighbors]

        return [row[1] for row in top_k], [row[0] for row in top_k]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for row in X_test:
            distances, neighbor_indices = self.kneighbors(row)
            neighbors_unique_classes, neighbors_unique_classes_counts = myutils.find_col_frequencies(self.y_train, neighbor_indices)
            winning_label = myutils.find_most_frequent_col_val(neighbors_unique_classes, neighbors_unique_classes_counts)
            y_predicted.append(winning_label)
            
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        unique_classes, unique_classes_counts = myutils.find_col_frequencies(y_train)
        self.most_common_label = myutils.find_most_frequent_col_val(unique_classes, unique_classes_counts)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i in range(0, len(X_test)):
            y_predicted.append(self.most_common_label)
        
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        labels, label_freqs = myutils.find_col_frequencies(y_train)
        for i in range(0, len(labels)):
            self.priors[labels[i]] = label_freqs[i] / len(y_train)

        self.posteriors = {}
        for i in range(0, len(X_train[0])):
            col = [X_train[j][i] for j in range(0, len(X_train))]
            unique_att_vals = myutils.get_unique_col_vals(col)
            for val in unique_att_vals:
                self.posteriors["att" + str(i+1) + "=" + str(val)] = {}
                for label in labels:
                    self.posteriors["att" + str(i+1) + "=" + str(val)][label] = 0
        
        for i in range(0, len(X_train)):
            for j in range(0, len(X_train[i])):
                self.posteriors["att" + str(j+1) + "=" + str(X_train[i][j])][y_train[i]] += (1 / label_freqs[labels.index(y_train[i])])

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        labels = list(self.priors.keys())
        for i in range(0, len(X_test)):
            pred_posteriors = [1 for _ in range(0, len(labels))]
            for j in range(0, len(labels)):
                for k in range(0, len(X_test[i])):
                    pred_posteriors[j] *= self.posteriors["att" + str(k+1) + "=" + str(X_test[i][k])][labels[j]]
                pred_posteriors[j] *= self.priors[labels[j]]
            y_predicted.append(labels[pred_posteriors.index(max(pred_posteriors))])

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = []
        self.attribute_domains = {}

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        header, attribute_domains = self.find_attribute_domains(self.X_train)
        self.header = header
        self.attribute_domains = attribute_domains
        train = [list(self.X_train[i]) + [self.y_train[i]] for i in range(len(self.X_train))]
        self.tree = self.tdidt(train, copy.deepcopy(self.header))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i in range(0, len(X_test)):
            subtree = copy.deepcopy(self.tree)
            while subtree[0] != "Leaf":
                if subtree[0] == "Attribute":
                    att_index = self.header.index(subtree[1])
                    for j in range(2, len(subtree)):
                        if subtree[j][1] == X_test[i][att_index]:
                            subtree = copy.deepcopy(subtree[j][2])
                            break
            y_predicted.append(subtree[1])
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names == None:
            attribute_names = self.header
        rule_start = "IF " + attribute_names[self.header.index(self.tree[1])] + " == "
        for i in range(2, len(self.tree)):
            subtree = self.tree[i]
            rule = rule_start + str(subtree[1])
            self.traverse_tree(copy.deepcopy(subtree[2]), attribute_names, class_name, copy.deepcopy(rule))

    def traverse_tree(self, subtree, attribute_names, class_name, rule = ""):
        if subtree[0] == "Leaf":
            print(rule + " THEN " + class_name + " == " + str(subtree[1]))
        else:
            rule += (" AND " + attribute_names[self.header.index(subtree[1])])
            for i in range(2, len(subtree)):
                new_subtree = subtree[i]
                new_rule = copy.deepcopy(rule)
                new_rule += (" == " + str(subtree[i][1]))
                self.traverse_tree(copy.deepcopy(new_subtree[2]), attribute_names, class_name, copy.deepcopy(new_rule))

    def find_attribute_domains(self, table):
        header = []
        attribute_domains = {}
        for i in range(0, len(table[0])):
            col_name = "att" + str(i)
            header.append(col_name)
            col = [table[j][i] for j in range(0, len(table))]
            col_domain = myutils.get_unique_col_vals(col)
            col_domain.sort()
            attribute_domains[col_name] = col_domain
        return header, attribute_domains

    def select_attribute(self, instances, attributes):
        entropies = [0 for _ in range(0, len(attributes))]
        for i in range(0, len(attributes)):
            partitions = self.partition_instances(instances, attributes[i])
            val_entropies = {}
            for att_value, att_partition in partitions.items():
                val_entropies[att_value] = 0
                classes, class_counts = myutils.find_col_frequencies([att_partition[i][-1] for i in range(0, len(att_partition))])
                for k in range(0, len(class_counts)):
                    if class_counts[k] != 0:
                        frac = class_counts[k] / len(att_partition)
                        val_entropies[att_value] += -1 * (frac) * math.log(frac, 2)
            Enew = 0
            for att_val, att_entropy in val_entropies.items():
                Enew += (len(partitions[att_val]) / len(instances)) * att_entropy
            entropies[i] = Enew

        return attributes[entropies.index(min(entropies))]

    def partition_instances(self, instances, attribute):
        # this is a group by attribute domain
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains["att" + str(att_index)]
        # print("attribute domain:", att_domain)
        # lets use dictionaries
        partitions = {}
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def same_class_label(self, instances):
        first_label = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_label:
                return False
        # get here, all the same
        return True

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):
        # print("available attributes:", available_attributes)

        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        # print("splitting on:", split_attribute)
        available_attributes.remove(split_attribute)
        # cannot split on this attribute again in this branch of tree
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # print("partitions:", partitions)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]
            if len(att_partition) > 0 and self.same_class_label(att_partition):
                # print("CASE 1 all same class label")
                #    CASE 1: all class labels of the partition are the same
                # => make a leaf node
                value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)])
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # print("CASE 2 clash")
                #    CASE 2: no more attributes to select (clash)
                # => handle clash w/majority vote leaf node
                winner = self.find_majority_vote_winner(att_partition)
                value_subtree.append(["Leaf", winner, len(att_partition), len(current_instances)])
            elif len(att_partition) == 0:
                # print("CASE 3 empty partition")
                #    CASE 3: no more instances to partition (empty partition)
                # => backtrack and replace attribute node with majority vote leaf node
                winner = self.find_majority_vote_winner(current_instances)
                value_subtree = ["Leaf", winner, len(current_instances)]
            else: # none of the bases cases were true...recurse!!
                # print("Recursing!!!")
                subtree = self.tdidt(att_partition, copy.deepcopy(available_attributes))
                if (subtree[0] == "Leaf" and len(subtree) == 3):
                    subtree.append(len(current_instances))
                value_subtree.append(subtree)
            if value_subtree[0] == "Leaf":
                return value_subtree
            tree.append(value_subtree)
        return tree

    def find_majority_vote_winner(self, att_partition):
            col = [att_partition[i][-1] for i in range(0, len(att_partition))]
            unique_vals, unique_val_counts = myutils.find_col_frequencies(col)
            winners = []
            max_count = max(unique_val_counts)
            for i in range(0, len(unique_val_counts)):
                if unique_val_counts[i] == max_count:
                    winners.append(unique_vals[i])
            winners.sort()
            return winners[0]