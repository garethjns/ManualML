# %% Breast cancer
if __name__ == '__main__':
    import sklearn.datasets

    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    XTrain, XValid, YTrain, YValid = tts(
        x, y, test_size=0.25, random_state=512)

    mod = Tree(minData=10, maxDepth=6,
               dynamicBias=False, bias=0.5)
    mod = mod.fit(XTrain, YTrain)
    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)

    accuracy(yPredTrain, YTrain)
    accuracy(yPredValid, YValid)

# %% Generated

if __name__ == '__main__':
    nF = 30
    X, Y = data = mk(n_samples=600,
                     n_features=nF,
                     n_informative=20,
                     n_redundant=5,
                     n_repeated=0,
                     n_classes=2)
    X = pd.DataFrame(X, columns=['x' + str(x) for x in range(nF)])
    Y = pd.DataFrame(Y)

    XTrain, XValid, YTrain, YValid = tts(
        X, Y, test_size=0.2, random_state=48)

    mod = Tree(minData=10, maxDepth=10,
               dynamicBias=False, bias=0.5)
    mod = mod.fit(XTrain, YTrain)
    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)

    mod.accuracy(YTrain, yPredTrain)
    mod.accuracy(YValid, yPredValid)
