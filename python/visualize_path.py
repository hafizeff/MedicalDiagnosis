import json
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from joblib import load
import numpy as np


def visualize_decision_path(model, user_input, feature_names, class_names, output_file):
    """
    Visualizes the decision tree along with the decision path for the input features.
    """
    # Reshape the user_input to 2D array, as decision_path expects a 2D array
    user_input_reshaped = np.array(user_input).reshape(1, -1)

    # Get the decision path for the input
    node_indicator = model.decision_path(user_input_reshaped)
    leaf_id = model.apply(user_input_reshaped)[0]

    # Predict the diagnosis
    prediction = model.predict(user_input_reshaped)[0]
    diagnosis = class_names[int(prediction)]  # Convert prediction to diagnosis name

    # Get the indices of the nodes in the decision path
    path_node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    # Create a figure and plot the tree
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,  # Add class names to display diagnosis at each node
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax
    )

    # Highlight nodes in the decision path
    for node_id in path_node_indices:
        # Check if the node has a corresponding bounding box
        if node_id < len(ax.texts):  # Ensure the node index is within bounds
            bbox = ax.texts[node_id].get_bbox_patch()
            if bbox:  # Ensure the bbox is not None
                bbox.set_edgecolor("red")
                bbox.set_linewidth(2)

    # Annotate the leaf node with its id and diagnosis
    ax.annotate(
        f"Leaf: {leaf_id} - {diagnosis}",
        xy=(0.5, -0.1),
        xycoords="axes fraction",
        fontsize=12,
        ha="center",
        va="center",
    )

    # Save the figure to the output file
    plt.savefig(output_file)
    plt.close()

    # Return only the diagnosis
    return diagnosis


if __name__ == "__main__":
    # Load the best model metadata
    with open("../cpp/best_trained_model.json", "r") as f:
        model_data = json.load(f)

    model_type = model_data["model_type"]
    feature_names = model_data["feature_names"]
    class_names = [str(c) for c in model_data["classes"]]  # Convert class names to strings
    model_file = model_data["model_file"]

    # Load the best model
    model = load(f"../python/{model_file}")

    # Read user input
    with open("input_features.json", "r") as f:
        input_data = json.load(f)

    # Exclude the name from the input data
    user_input = input_data[0:]  # Skip the first element (name)

    # Encode "yes" and "no" to 1 and 0 for categorical features
    for i, feature in enumerate(user_input):
        if isinstance(feature, str):
            user_input[i] = 1.0 if feature.lower() == "yes" else 0.0

    # Ensure all inputs are floats (required by sklearn models)
    user_input = np.array([float(value) for value in user_input])

    # Ensure the number of features matches the expected count
    if len(user_input) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, but got {len(user_input)}. Please check the input data.")

    # Visualize the decision path and get the diagnosis
    if model_type == "DecisionTree":
        print(visualize_decision_path(model, user_input, feature_names, class_names, "decision_path.png"))
    elif model_type == "RandomForest":
        print(visualize_decision_path(model.estimators_[0], user_input, feature_names, class_names, "decision_path.png"))
