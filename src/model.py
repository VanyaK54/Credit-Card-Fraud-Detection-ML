from sklearn.ensemble import RandomForestClassifier

def train_model(xTrain, yTrain):
    model = RandomForestClassifier(random_state=42)
    model.fit(xTrain, yTrain)
    return model

def predict(model, xTest):
    return model.predict(xTest)
