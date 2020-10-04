def accuracy(y_true, y_pred):
    '''
    Function to calculate accuracy
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Accuracy score
    '''

    ctr = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            ctr += 1

    # Return accuracy, correct predictions over number of samples
    return ctr / len(y_true)

def true_positive(y_true, y_pred):
    '''
    Function to calculate true positive
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Number of true positives
    '''

    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    return tp

def true_negative(y_true, y_pred):
    '''
    Function to calculate true negative
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Number of true negatives
    '''

    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1

    return tn

def false_positive(y_true, y_pred):
    '''
    Function to calculate False positive
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Number of false positives
    '''

    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1

    return fp

def false_negative(y_true, y_pred):
    '''
    Function to calculate false negative
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Number of false negatives
    '''

    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1

    return fn

def accuracy_new(y_true, y_pred):
    '''
    Function to calculate accuracy using tp, tn, fp, fn
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Accuracy score
    '''

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples

    return accuracy

def precision(y_true, y_pred):
    '''
    Function to calculate precision
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Precision score
    '''

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision

def recall(y_true, y_pred):
    '''
    Function to calculate recall
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : Recall score
    '''

    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

def f1_score(y_true, y_pred):
    '''
    Function to calculate F1 score
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : F1 score
    '''

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    f1_score = 2 * p * r / (p + r)

    return f1_score

def true_positive_rate(y_true, y_pred):
    '''
    Function to calculate True Positive Rate, same as recall, also known as sensitivity
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : True Positive Rate
    '''

    tpr = recall(y_true, y_pred)

    return tpr

def false_positive_rate(y_true, y_pred):
    '''
    Function to calculate False Positive Rate, also known as specificity
    :param y_true : List of true values
    :param y_pred : List of predicted values
    :return : False Positive Rate
    '''

    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    fpr = fp / (fp + tn)

    return fpr
