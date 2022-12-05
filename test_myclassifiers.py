import numpy as np
from scipy import stats
from mysklearn.myclassifiers import MyNaiveBayesClassifier, \
    MyKNeighborsClassifier,\
    MyDummyClassifier, MyDecisionTreeClassifier

def test_kneighbors_classifier_kneighbors():
    knn_clf1 = MyKNeighborsClassifier(3)
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    knn_clf1.fit(X_train_class_example1, y_train_class_example1)
    distances, neighbor_indices = knn_clf1.kneighbors([0.33, 1])
    rounded_distances = [round(val, 2) for val in distances]
    expected_distances = [2/3, 1.0, ((((1/3) ** 2) + 1) ** (1/2))]
    rounded_expected_distances = [round(val, 2) for val in expected_distances]
    assert np.allclose(rounded_distances, rounded_expected_distances)
    assert neighbor_indices == [0, 2, 3]

    knn_clf2 = MyKNeighborsClassifier(3)
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn_clf2.fit(X_train_class_example2, y_train_class_example2)
    distances, neighbor_indices = knn_clf2.kneighbors([2, 3])
    rounded_distances = [round(val, 2) for val in distances]
    expected_distances = [(2 ** (1/2)), (2 ** (1/2)), 2.0]
    rounded_expected_distances = [round(val, 2) for val in expected_distances]
    assert np.allclose(rounded_distances, rounded_expected_distances)
    assert neighbor_indices == [0, 4, 6]

    knn_clf3 = MyKNeighborsClassifier(5)
    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    knn_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    distances, neighbor_indices = knn_clf3.kneighbors([9.1, 11.0])
    rounded_distances = [round(val, 3) for val in distances]
    assert np.allclose(rounded_distances, [0.608, 1.237, 2.202, 2.802, 2.915])
    assert neighbor_indices == [6, 5, 7, 4, 8]

def test_kneighbors_classifier_predict():
    knn_clf1 = MyKNeighborsClassifier(3)
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    knn_clf1.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = knn_clf1.predict([[0.33, 1]])
    assert y_predicted == ["good"]

    knn_clf2 = MyKNeighborsClassifier(3)
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn_clf2.fit(X_train_class_example2, y_train_class_example2)
    y_predicted = knn_clf2.predict([[2, 3]])
    assert y_predicted == ["yes"]

    knn_clf3 = MyKNeighborsClassifier(5)
    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    knn_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    y_predicted = knn_clf3.predict([[9.1, 11.0]])
    assert y_predicted == ["+"]

def test_dummy_classifier_fit():
    np.random.seed(0)
    dummy_clf1 = MyDummyClassifier()
    X_train = [[0] for _ in range(0, 100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf1.fit(X_train, y_train)
    assert dummy_clf1.most_common_label == "yes"

    dummy_clf2 = MyDummyClassifier()
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf2.fit(X_train, y_train)
    assert dummy_clf2.most_common_label == "no"

    dummy_clf3 = MyDummyClassifier()
    y_train = list(np.random.choice(["yes", "no", "maybe", "idk"], 100, replace=True, p=[0.1, 0.1, 0.2, 0.6]))
    dummy_clf3.fit(X_train, y_train)
    assert dummy_clf3.most_common_label == "idk"

def test_dummy_classifier_predict():
    np.random.seed(0)
    dummy_clf1 = MyDummyClassifier()
    X_train = [[0] for _ in range(0, 100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf1.fit(X_train, y_train)
    y_predicted = dummy_clf1.predict([[0]])
    assert y_predicted == ["yes"]

    dummy_clf2 = MyDummyClassifier()
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf2.fit(X_train, y_train)
    y_predicted = dummy_clf2.predict([[0]])
    assert y_predicted == ["no"]

    dummy_clf3 = MyDummyClassifier()
    y_train = list(np.random.choice(["yes", "no", "maybe", "idk"], 100, replace=True, p=[0.1, 0.1, 0.2, 0.6]))
    dummy_clf3.fit(X_train, y_train)
    y_predicted = dummy_clf3.predict([[0], [12]])
    assert y_predicted == ["idk", "idk"]

def test_naive_bayes_classifier_fit():
    nb_clf1 = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    nb_clf1.fit(X_train_inclass_example, y_train_inclass_example)
    expected_priors = {"yes": 5/8, "no": 3/8}
    priors_keys = ["yes", "no"]
    for key in priors_keys:
        assert np.isclose(nb_clf1.priors[key], expected_priors[key])
    expected_posteriors = {"att1=1": {"yes": 4/5, "no": 2/3}, 
                           "att1=2": {"yes": 1/5, "no": 1/3}, 
                           "att2=5": {"yes": 2/5, "no": 2/3},
                           "att2=6": {"yes": 3/5, "no": 1/3}}
    posteriors_keys = ["att1=1", "att1=2", "att2=5", "att2=6"]
    for post_key in posteriors_keys:
        for pre_key in priors_keys:
            assert np.isclose(nb_clf1.posteriors[post_key][pre_key], expected_posteriors[post_key][pre_key])

    nb_clf2 = MyNaiveBayesClassifier()
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    nb_clf2.fit(X_train_iphone, y_train_iphone)
    expected_priors = {"yes": 10/15, "no": 5/15}
    priors_keys = ["yes", "no"]
    for key in priors_keys:
        assert np.isclose(nb_clf2.priors[key], expected_priors[key])
    expected_posteriors = {"att1=1": {"yes": 2/10, "no": 3/5},
                           "att1=2": {"yes": 8/10, "no": 2/5},
                           "att2=1": {"yes": 3/10, "no": 1/5},
                           "att2=2": {"yes": 4/10, "no": 2/5},
                           "att2=3": {"yes": 3/10, "no": 2/5},
                           "att3=fair": {"yes": 7/10, "no": 2/5},
                           "att3=excellent": {"yes": 3/10, "no": 3/5}}
    posteriors_keys = ["att1=1", "att1=2", "att2=1", "att2=2", "att2=3", "att3=fair", "att3=excellent"]
    for post_key in posteriors_keys:
        for pre_key in priors_keys:
            assert np.isclose(nb_clf2.posteriors[post_key][pre_key], expected_posteriors[post_key][pre_key])

    nb_clf3 = MyNaiveBayesClassifier()
    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    nb_clf3.fit(X_train_train, y_train_train)
    expected_priors = {"on time": 14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20}
    priors_keys = ["on time", "late", "very late", "cancelled"]
    for key in priors_keys:
        assert np.isclose(nb_clf3.priors[key], expected_priors[key])
    expected_posteriors = {"att1=weekday": {"on time": 9/14, "late": 1/2, "very late": 3/3, "cancelled": 0/1},
                           "att1=saturday": {"on time": 2/14, "late": 1/2, "very late": 0/3, "cancelled": 1/1},
                           "att1=sunday": {"on time": 1/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1},
                           "att1=holiday": {"on time": 2/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1},
                           "att2=spring": {"on time": 4/14, "late": 0/2, "very late": 0/3, "cancelled": 1/1},
                           "att2=summer": {"on time": 6/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1},
                           "att2=autumn": {"on time": 2/14, "late": 0/2, "very late": 1/3, "cancelled": 0/1},
                           "att2=winter": {"on time": 2/14, "late": 2/2, "very late": 2/3, "cancelled": 0/1},
                           "att3=none": {"on time": 5/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1},
                           "att3=high": {"on time": 4/14, "late": 1/2, "very late": 1/3, "cancelled": 1/1},
                           "att3=normal": {"on time": 5/14, "late": 1/2, "very late": 2/3, "cancelled": 0/1},
                           "att4=none": {"on time": 5/14, "late": 1/2, "very late": 1/3, "cancelled": 0/1},
                           "att4=slight": {"on time": 8/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1},
                           "att4=heavy": {"on time": 1/14, "late": 1/2, "very late": 2/3, "cancelled": 1/1}}
    posteriors_keys = ["att1=weekday", "att1=saturday", "att1=sunday", "att1=holiday",
                       "att2=spring", "att2=summer", "att2=autumn", "att2=winter",
                       "att3=none", "att3=high", "att3=normal",
                       "att4=none", "att4=slight", "att4=heavy"]
    for post_key in posteriors_keys:
        for pre_key in priors_keys:
            assert np.isclose(nb_clf3.posteriors[post_key][pre_key], expected_posteriors[post_key][pre_key])
    
def test_naive_bayes_classifier_predict():
    nb_clf1 = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    nb_clf1.fit(X_train_inclass_example, y_train_inclass_example)
    y_predicted = nb_clf1.predict([[1, 5]])
    assert y_predicted == ["yes"]

    nb_clf2 = MyNaiveBayesClassifier()
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    nb_clf2.fit(X_train_iphone, y_train_iphone)
    y_predicted = nb_clf2.predict([[2, 2, "fair"], [1, 1, "excellent"]])
    assert y_predicted == ["yes", "no"]

    nb_clf3 = MyNaiveBayesClassifier()
    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    nb_clf3.fit(X_train_train, y_train_train)
    y_predicted = nb_clf3.predict([["weekday", "winter", "high", "heavy"], 
                                   ["weekday", "summer", "high", "heavy"], 
                                   ["sunday", "summer", "normal", "slight"]])
    assert y_predicted == ["very late", "on time", "on time"]

def test_decision_tree_classifier_fit():
    tree_clf1 = MyDecisionTreeClassifier()
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree_clf1.fit(X_train_interview, y_train_interview)
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    tree_clf1.print_decision_rules(header_interview, "interviewed_well")
    assert tree_clf1.tree == tree_interview

    tree_clf2 = MyDecisionTreeClassifier()
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    tree_clf2.fit(X_train_iphone, y_train_iphone)
    tree_iphone = \
            ["Attribute", "att0",
                ["Value", 1,
                    ["Attribute", "att1",
                        ["Value", 1,
                            ["Leaf", "yes", 1, 5]
                        ],
                        ["Value", 2, 
                            ["Attribute", "att2",
                                ["Value", "excellent",
                                    ["Leaf", "yes", 1, 2]
                                ],
                                ["Value", "fair",
                                    ["Leaf", "no", 1, 2]
                                ]
                            ]
                        ],
                        ["Value", 3,
                            ["Leaf", "no", 2, 5]
                        ]
                    ]
                ],
                ["Value", 2,
                    ["Attribute", "att2",
                        ["Value", "excellent",
                            ["Leaf", "no", 4, 10]
                        ],
                        ["Value", "fair",
                            ["Leaf", "yes", 6, 10]
                        ]
                    ]
                ]
            ]
    tree_clf2.print_decision_rules(header_iphone, "buys_iphone")
    assert tree_clf2.tree == tree_iphone

def test_decision_tree_classifier_predict():
    tree_clf1 = MyDecisionTreeClassifier()
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree_clf1.fit(X_train_interview, y_train_interview)
    y_predicted = tree_clf1.predict([["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]])
    assert y_predicted == ["True", "False"]

    tree_clf2 = MyDecisionTreeClassifier()
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    tree_clf2.fit(X_train_iphone, y_train_iphone)
    y_predicted = tree_clf2.predict([[2, 2, "fair"], [1, 1, "excellent"]])
    assert y_predicted == ["yes", "yes"]