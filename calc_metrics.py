from sklearn.metrics import accuracy_score, precision_score, recall_score


def calc_metrics(y_train, y_test, y_pre_train, y_pre_test):
    acc_train = accuracy_score(y_train, y_pre_train)
    acc_test = accuracy_score(y_test, y_pre_test)
    pr_test = precision_score(y_test, y_pre_test)
    rc_test = recall_score(y_test, y_pre_test)

    return acc_train, acc_test, pr_test, rc_test
