from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC

def missing_values_fillin(data, column, fillin):
    # Can fill in some string value or mean of data
    data.isna().sum()
    data.fillna(fillin, inplace=True)
    data.isna().sum()
    len(data)
    return data

def missing_values_drop(data):
    data.dropna(inplace=True)
    len(data)
    return data

def missing_values_fillin_with_scikit(data):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    # Fill categorical values with 'missing' & numerical values with mean
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    door_imputer = SimpleImputer(strategy="constant", fill_value=4)
    num_imputer = SimpleImputer(strategy="mean")

    # Define columns
    cat_features = ["Make", "Colour"]
    door_feature = ["Doors"]
    num_features = ["Odometer (KM)"]

    # Create an imputer (something that fills missing data)
    imputer = ColumnTransformer([
        ("cat_imputer", cat_imputer, cat_features),
        ("door_imputer", door_imputer, door_feature),
        ("num_imputer", num_imputer, num_features)
    ])

    # Fill train and test values separately
    filled_X_train = imputer.fit_transform(X_train)
    filled_X_test = imputer.transform(X_test)

    #return filled_X_train, filled_X_test
    # Get our transformed data array's back into DataFrame's
    car_sales_filled_train = pd.DataFrame(filled_X_train, 
                                      columns=["Make", "Colour", "Doors", "Odometer (KM)"])

    car_sales_filled_test = pd.DataFrame(filled_X_test, 
                                     columns=["Make", "Colour", "Doors", "Odometer (KM)"])

def eda_on_data(data, target):
    data.head()
    data.describe()
    data.dtypes
    X = data.drop(target, axis=1)
    X.head()
    print("Y column ")
    y = data[target]
    y.head()
    len(data)
    data[col].value_counts()

def split_data(test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape     
    return X_train, X_test, y_train, y_test

def one_hot_encoder(cat_features)
    # Categorical features as list
    # categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                   one_hot,
                                   cat_features)],
                                   remainder="passthrough")

    transformed_X = transformer.fit_transform(X)
    pd.DataFrame(transformed_X)
    return transformed_X

def fit_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def fit_regression_model(model, X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    
    # Instatiate Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    model.score(X_test, y_test)

def predict_regression(model, X_test, y_test):
    # Make predictions
    y_preds = model.predict(X_test)
    # Compare the predictions to the truth
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error(y_test, y_preds)

def fit_classification_model(clf, X_train, y_train):
    # Instantiate LinearSVC
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # Instantiate Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

def predict(clf, X_test, y_test):
    from sklearn.metrics import accuracy_score
    y_preds = clf.predict(X_test)
    # Compare predictions to truth labels to evaluate the model
    y_preds = clf.predict(X_test)
    np.mean(y_preds == y_test)
    accuracy_score(y_test, y_preds)

def Predict_probability(clf, X_test):
    # predict_proba() returns probabilities of a classification label 
    clf.predict_proba(X_test[:5])

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    # Score method
    clf.score(X_train, y_train)
    clf.score(X_test, y_test)
    
    # Evaluating using scoring parameter
    # # Take the mean of 5-fold cross-validation score
    cross_val_score(clf, X, y, cv=5)
    cross_val_score(clf, X, y, cv=10)
    # Scoring parameter set to None by default
    cross_val_score(clf, X, y, cv=5, scoring=None)
    
   
def save_model():
    
def load_model():
 
# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestRegressor())])
