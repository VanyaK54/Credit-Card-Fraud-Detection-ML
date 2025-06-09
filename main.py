from src import preprocess, model, evaluate, utils

# Load and preprocess data
data = preprocess.load_data('data/creditcard.csv')
X, Y = preprocess.preprocess_data(data)
xTrain, xTest, yTrain, yTest = preprocess.split_and_resample(X, Y)

# Train model and make predictions
clf = model.train_model(xTrain, yTrain)
yPred = model.predict(clf, xTest)

# Evaluate and plot
metrics = evaluate.evaluate_model(yTest, yPred)
for key, value in metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

evaluate.plot_confusion_matrix(yTest, yPred, 'outputs/figures/confusion_matrix.png')
utils.plot_correlation_matrix(data, 'outputs/figures/correlation_matrix.png')
