import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Added for train-test split
import joblib
import json
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def train_and_evaluate_model(input_file, model_metadata_file):
    # Manually define the feature names
    feature_names = ["age", "gender", "fever", "cough", "headache", "fatigue", "breathlessness"]

    # Load data, skipping the first row (which contains column names)
    data = pd.read_csv(input_file, header=None, skiprows=1)

    # Set the column names manually
    data.columns = feature_names + ['diagnosis']  # Assuming the last column is the diagnosis

    # Separate features and labels
    X = data[feature_names]  # Features (all columns except diagnosis)
    y = data['diagnosis']    # Label (last column: diagnosis)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train decision tree classifier
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)

    # Visualize and save the decision tree as an image
    dot_data = export_graphviz(
        dt_clf, 
        out_file=None, 
        feature_names=feature_names, 
        filled=True, 
        rounded=True, 
        special_characters=True
    )
    dt_graph = graphviz.Source(dot_data)
    dt_graph.render("decision_tree", format="png", cleanup=True)  # Save as PNG

    # Train random forest classifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    # Visualize and save the first decision tree of the random forest
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_tree(
        rf_clf.estimators_[0], 
        filled=True, 
        feature_names=feature_names, 
        class_names=rf_clf.classes_, 
        rounded=True, 
        ax=ax
    )
    plt.savefig("random_forest_tree.png")  # Save as PNG

    # Get accuracy for both models on the test set
    dt_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))
    rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))

    print(f"Decision Tree Test Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")

    # Select the best model based on accuracy
    if dt_accuracy > rf_accuracy:
        best_model = "DecisionTree"
        best_model_file = "decision_tree_model.joblib"
        joblib.dump(dt_clf, best_model_file)
        best_accuracy = dt_accuracy
    else:
        best_model = "RandomForest"
        best_model_file = "random_forest_model.joblib"
        joblib.dump(rf_clf, best_model_file)
        best_accuracy = rf_accuracy

    # Prepare metadata for the best model
    best_model_info = {
        "model_type": best_model,
        "accuracy": best_accuracy,
        "model_file": best_model_file,
        "feature_names": feature_names,
        "classes": dt_clf.classes_.tolist() if best_model == "DecisionTree" else rf_clf.classes_.tolist()
    }

    # Save metadata to JSON
    with open(model_metadata_file, "w") as f:
        json.dump(best_model_info, f)

    print(f"The best model ({best_model}) has been saved to {best_model_file}")
    print(f"Best model information saved to {model_metadata_file}")

if __name__ == "__main__":
    train_and_evaluate_model("../python/processed_data.csv", "../cpp/best_trained_model.json")
