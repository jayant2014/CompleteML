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
    Function to calculate accurayc using tp, tn, fp, fn
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
