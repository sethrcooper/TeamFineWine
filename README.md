# Team Fine Wine

```
pip install -r requirements.txt
```

## Naive Bayes

### Training

1. Navigate to naive_bayes_test.ipynb
2. Run all cells

### Testing

1. Navigate to WineClassifier.py
2. Execute WineClassifier.py
3. View results in console

## CNN

Utilize Python 3.10.6

### Training

1. Navigate to cnn/cnn.ipynb
2. Run all cells

### Testing

1. Navigate to WineClassifier.py
2. Execute WineClassifier.py
3. View results in console

## Multiclass Logistic Regression

### Training

1. Navigate to WineClassifier.py
2. Ensure line 63 containing the function multi_logistic_regression is uncommented
3. Ensure line 65 containing the function predictPreset is commented out
4. Execute WineClassifier.py
5. View results in console

### Testing

1. Navigate to WineClassifier.py
2. Ensure line 63 containing the function multi_logistic_regression is commented out
3. Ensure line 65 containing the function predictPreset is uncommented
4. Execute WineClassifier.py
5. View results in console

The function multi_logistic_regression trains a new weight vector for each random training-testing split each time WineClassifier.py is run and tests it on the testing subset

The function predictPreset makes predictions given X_test using a static weight vector pulled from a single run that averaged high results.
