def accuracy(y_true, y_pred):
    '''
    Function to calculate accuracy
    :param y_true : List of true values
    :param y_pred : list of predicted values
    :return : accuracy score
    '''
    ctr = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            ctr += 1

    # Return accuracy, correct predictions over number of samples
    return ctr / len(y_true)
