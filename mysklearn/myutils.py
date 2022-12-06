import numpy as np # use numpy's random number generation
import csv

def compute_euclidean_distance(v1, v2):
    total = 0
    for i in range(len(v1)):
        if (type(v1[i]) is int or type(v1[i]) is float) and (type(v2[i]) is int or type(v2[i]) is float):
            total += (v1[i] - v2[i]) ** 2
        elif v1 == v2:
            total += 0
        elif v1 != v2:
            total += 1
    return np.sqrt(total)

def find_col_frequencies(col, indices=None):
    unique_vals = []
    unique_val_counts = []
    if indices == None:
        for i in range(0, len(col)):
            if col[i] in unique_vals:
                unique_val_counts[unique_vals.index(col[i])] += 1
            else:
                unique_vals.append(col[i])
                unique_val_counts.append(1)
    else:
        for i in range(0, len(indices)):
            if col[indices[i]] in unique_vals:
                unique_val_counts[unique_vals.index(col[indices[i]])] += 1
            else:
                unique_vals.append(col[indices[i]])
                unique_val_counts.append(1)

    return unique_vals, unique_val_counts

def find_most_frequent_col_val(unique_vals, unique_val_counts):
    max_val_count = 0
    for i in range(0, len(unique_vals)):
        if unique_val_counts[i] > max_val_count:
            max_val_count = unique_val_counts[i]
            winning_val = unique_vals[i]

    return winning_val

def convert_to_numeric(values):
    for i in range(len(values)):
        try:
            if ("." in values[i]):
                numeric_val = float(values[i])
            else:
                numeric_val = int(values[i])
            values[i] = numeric_val
        except ValueError as e:
            pass

def read_table(filename):
    table = []
    
    infile = open(filename, "r") # open is a function

    reader = csv.reader(infile)
    for row in reader:
        convert_to_numeric(row)
        table.append(row)

    infile.close() # close is a method

    return table

def find_doe_mpg_rating(mpg):
    if mpg < 13.5:
        return "1"
    if mpg >= 13.5 and mpg < 14.5:
        return "2"
    if mpg >= 14.5 and mpg < 16.5:
        return "3"
    if mpg >= 16.5 and mpg < 19.5:
        return "4"
    if mpg >= 19.5 and mpg < 23.5:
        return "5"
    if mpg >= 23.5 and mpg < 26.5:
        return "6"
    if mpg >= 26.5 and mpg < 30.5:
        return "7"
    if mpg >= 30.5 and mpg < 36.5:
        return "8"
    if mpg >= 36.5 and mpg < 44.5:
        return "9"
    if mpg >= 44.5:
        return "10"

def determine_class_label(tm_pts, opp_pts):
    if tm_pts - opp_pts >= 20:
        return "BlowoutW"
    elif tm_pts - opp_pts >= 11:
        return "ComfortableW"
    elif tm_pts - opp_pts > 0:
        return "CloseW"
    else:
        return "L"

def determine_loc(given_loc):
    if given_loc == "@":
        return "A"
    elif given_loc == "N":
        return "N"
    else:
        return "H"

def discretize_fgs(fg, fga, fgp):
    fg_d = ""
    fga_d = ""
    fgp_d = ""
    if fg >= 35:
        fg_d = "High"
    elif fg >= 25:
        fg_d = "Average"
    else:
        fg_d = "Low"
    if fga >= 70:
        fga_d = "High"
    elif fga >= 55:
        fga_d = "Average"
    else:
        fga_d = "Low"
    if fgp >= 0.55:
        fgp_d = "High"
    elif fgp >= 0.5:
        fgp_d = "Average"
    else:
        fgp_d = "Low"
    return fg_d, fga_d, fgp_d

def discretize_twos(two, twoa, twop):
    two_d = ""
    twoa_d = ""
    twop_d = ""
    if two >= 30:
        two_d = "High"
    elif two >= 20:
        two_d = "Average"
    else:
        two_d = "Low"
    if twoa >= 45:
        twoa_d = "High"
    elif twoa >= 30:
        twoa_d = "Average"
    else:
        twoa_d = "Low"
    if twop >= 0.6:
        twop_d = "High"
    elif twop >= 0.5:
        twop_d = "Average"
    else:
        twop_d = "Low"
    return two_d, twoa_d, twop_d

def discretize_threes(three, threea, threep):
    three_d = ""
    threea_d = ""
    threep_d = ""
    if three >= 10:
        three_d = "High"
    elif three >= 6:
        three_d = "Average"
    else:
        three_d = "Low"
    if threea >= 25:
        threea_d = "High"
    elif threea >= 15:
        threea_d = "Average"
    else:
        threea_d = "Low"
    if threep >= 0.4:
        threep_d = "High"
    elif threep >= 0.3:
        threep_d = "Average"
    else:
        threep_d = "Low"
    return three_d, threea_d, threep_d

def discretize_fts(ft, fta, ftp):
    ft_d = ""
    fta_d = ""
    ftp_d = ""
    if ft >= 15:
        ft_d = "High"
    elif ft >= 10:
        ft_d = "Average"
    else:
        ft_d = "Low"
    if fta >= 25:
        fta_d = "High"
    elif fta >= 15:
        fta_d = "Average"
    else:
        fta_d = "Low"
    if ftp >= 0.8:
        ftp_d = "High"
    elif ftp >= 0.6:
        ftp_d = "Average"
    else:
       ftp_d = "Low"
    return ft_d, fta_d, ftp_d

def find_accuracy(y_predicted, y_test):
    correct_preds = 0
    for i in range(0, len(y_test)):
        if y_predicted[i] == y_test[i]:
            correct_preds += 1
    return correct_preds / len(y_predicted)

def print_classification_results(y_predicted, y_test, test_set, name):
    print("===========================================")
    print(name)
    print("===========================================")
    for i in range(0, len(test_set)):
        print("instance: ", test_set[i])
        print("class: ", y_predicted[i], " actual: ", y_test[i])
    print("accuracy: ", find_accuracy(y_predicted, y_test))

def normalize(col, min_val=None, max_val=None):
    if min_val == None and max_val == None:
        return [(col[i] - min(col)) / ((max(col) - min(col)) * 1.0) for i in range(0, len(col))], min(col), max(col)
    elif type(col) != list:
        return (col - min_val) / ((max_val- min_val) * 1.0)
    else:
        return [(col[i] - min_val) / ((max_val- min_val) * 1.0) for i in range(0, len(col))]

def shuffle_instances(X, y=None, random_state=0):
    np.random.seed(random_state)
    X_shuffled = []
    y_shuffled = []

    while(len(X) > 0):
        index = np.random.randint(0, len(X))
        X_shuffled.append(X.pop(index))
        if y != None:
            y_shuffled.append(y.pop(index))

    if y != None:
        return X_shuffled, y_shuffled
    else:
        return X_shuffled

def get_unique_col_vals(col):
    unique_col_vals = []
    for val in col:
        if val not in unique_col_vals:
            unique_col_vals.append(val)

    return unique_col_vals