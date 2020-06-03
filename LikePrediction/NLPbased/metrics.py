from keras import backend


# Calculate recall
def recall(y_test, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_test * y_pred, 0, 1)))
    positives = backend.sum(backend.round(backend.clip(y_test, 0, 1)))
    return tp / (positives + backend.epsilon())


# Calculate precision
def precision(y_test, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_test * y_pred, 0, 1)))
    positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    return tp / (positives + backend.epsilon())


# Calculate f1
def f1(y_test, y_pred):
    pr = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    return 2*((pr*rec)/(pr+rec+backend.epsilon()))
