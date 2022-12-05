from math import ceil
import numpy as np # use numpy's random number generation
import copy

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if type(test_size) == int:
        num_test_set_instances = test_size
    elif type(test_size) == float:
        num_test_set_instances = ceil(len(X) * test_size)

    X_copy = copy.deepcopy(X)
    y_copy = copy.deepcopy(y)
    if shuffle:
        X_copy, y_copy = myutils.shuffle_instances(X_copy, y_copy, random_state)

    X_test = [[] for _ in range(0, num_test_set_instances)]
    y_test = [[] for _ in range(0, num_test_set_instances)]
    for i in range(0, num_test_set_instances):
        X_test[len(X_test) - i - 1] = X_copy.pop(len(X_copy) - 1)
        y_test[len(y_test) - i - 1] = y_copy.pop(len(y_copy) - 1)

    return X_copy, X_test, y_copy, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    list_of_indices = [i for i in range(0, len(X))]
    if shuffle:
        list_of_indices = myutils.shuffle_instances(list_of_indices, random_state=random_state)

    k_folds = [[] for _ in range(0, n_splits)]
    for i in range(0, len(list_of_indices)):
        k_folds[i % n_splits].append(list_of_indices[i])

    X_train_test = [[] for _ in range(0, n_splits)]
    for i in range(0, n_splits):
        k_folds_copy = copy.deepcopy(k_folds)
        test_fold = k_folds_copy.pop(i)
        X_train_test[i] = [item for fold in k_folds_copy for item in fold], test_fold

    return X_train_test

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    class_labels = myutils.get_unique_col_vals(y)
    class_labels_indices = [[] for _ in range(0, len(class_labels))]
    list_of_indices = [i for i in range(0, len(X))]
    if shuffle:
        list_of_indices = myutils.shuffle_instances(list_of_indices, random_state=random_state)
    
    for i in range(0, len(class_labels)):
        indices_list = []
        for j in range(0, len(list_of_indices)):
            if y[list_of_indices[j]] == class_labels[i]:
                indices_list.append(list_of_indices[j])
        class_labels_indices[i] = indices_list

    k_folds = [[] for _ in range(0, n_splits)]
    for i in range(0, len(class_labels_indices)):
        for j in range(0, len(class_labels_indices[i])):
            k_folds[j % n_splits].append(class_labels_indices[i][j])

    X_train_test = [[] for _ in range(0, n_splits)]
    for i in range(0, n_splits):
        k_folds_copy = copy.deepcopy(k_folds)
        test_fold = k_folds_copy.pop(i)
        X_train_test[i] = [item for fold in k_folds_copy for item in fold], test_fold
    
    return X_train_test

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state == None:
        random_state = 0
    np.random.seed(random_state)
    X_sample = []
    y_sample = []
    selected_indices = []
    if n_samples == None:
        n_samples = len(X)

    for _ in range(0, n_samples):
        index = np.random.randint(0, len(X))
        selected_indices.append(index)
        X_sample.append(X[index])
        if y is not None:
            y_sample.append(y[index])

    X_out_of_bag = []
    y_out_of_bag = []
    for i in range(0, len(X)):
        if i not in selected_indices:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    if y is not None:
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag
    else:
        return X_sample, X_out_of_bag, None, None

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    confusion_matrix = [[0 for _ in labels] for _ in labels]

    for i in range(0, len(y_true)):
        confusion_matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    return confusion_matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = 0
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    
    if not normalize:
        return num_correct
    if normalize:
        return num_correct / len(y_true)

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    confusing_matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label == None:
        pos_index = 0
        neg_index = 1
    else:
        pos_index = labels.index(pos_label)
        if pos_index == 0:
            neg_index = 1
        if pos_index == 1:
            neg_index = 0
    
    tp = confusing_matrix[pos_index][pos_index]
    fp = confusing_matrix[neg_index][pos_index]

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    confusing_matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label == None:
        pos_index = 0
        neg_index = 1
    else:
        pos_index = labels.index(pos_label)
        if pos_index == 0:
            neg_index = 1
        if pos_index == 1:
            neg_index = 0

    tp = confusing_matrix[pos_index][pos_index]
    fn = confusing_matrix[pos_index][neg_index]

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)