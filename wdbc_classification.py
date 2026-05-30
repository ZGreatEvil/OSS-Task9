import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, tree, neighbors, ensemble, metrics
from matplotlib.lines import Line2D


# step 2: load the dataset

# load the breast cancer dataset from scikit-learn
wdbc = datasets.load_breast_cancer()

# store input features and target labels
X = wdbc.data
y = wdbc.target

# print dataset information
print("Data shape:", X.shape) # 569 samples, 30 features
print("Target shape:", y.shape) # 569 labels
print("Target names:", wdbc.target_names) # malignant, benign
print("First 5 feature names:", wdbc.feature_names[:5])


# step 3: train the default svm classifier

# create and train the default svm model
model = svm.SVC()
model.fit(X, y)
predict = model.predict(X)

# calculate evaluation metrics for the default svm
accuracy  = metrics.accuracy_score(y, predict)
precision = metrics.precision_score(y, predict)
recall    = metrics.recall_score(y, predict)

print("\n--- default svm classifier ---")
print("Accuracy: ", round(accuracy,  4))
print("Precision:", round(precision, 4))
print("Recall:   ", round(recall,    4))


# step 4: train and compare multiple classifiers

# define all classifiers to compare
models = {
    "SVM":           svm.SVC(),
    "Decision Tree": tree.DecisionTreeClassifier(random_state=42),
    "KNN":           neighbors.KNeighborsClassifier(),
    "Random Forest": ensemble.RandomForestClassifier(random_state=42),
}

# train each classifier and print its accuracy
print("\n--- classifier comparison ---")
results = {}
for name, clf in models.items():
    clf.fit(X, y)
    pred = clf.predict(X)
    acc  = metrics.accuracy_score(y, pred)
    results[name] = {"model": clf, "pred": pred, "accuracy": acc}
    print(f"{name} Accuracy: {round(acc, 4)}")


# step 5: find the best classifier 

# find the classifier with the highest accuracy
best_name = max(results, key=lambda n: results[n]["accuracy"])
best_acc  = results[best_name]["accuracy"]
best_pred = results[best_name]["pred"]

print(f"\nbest classifier: {best_name} (accuracy: {round(best_acc, 4)})")


# step 6: confusion matrix for the best classifier 

# create confusion matrix using the best model's predictions
conf_matrix = metrics.confusion_matrix(y, best_pred)

# display and save the confusion matrix
disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=wdbc.target_names,
)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("wdbc_classification_matrix.png", dpi=150)
plt.close()
print("saved to 'wdbc_classification_matrix.png'")


# step 7: scatter plot 

# use mean radius (feature 0) and mean texture (feature 1) for the scatter plot
# these are the first two features and easy to interpret visually
x_idx = 0
y_idx = 1

# define colors for the two classes
# malignant = red, benign = blue
colors = np.array([
    (0.85, 0.15, 0.15), # red for malignant
    (0.20, 0.45, 0.85), # blue for benign
])

# build legend entries
legend_handles = [
    Line2D([0], [0], marker="o", lw=0, label=wdbc.target_names[i], color=colors[i])
    for i in range(len(colors))
]

# create the scatter plot colored by true class with edge color = predicted class
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, x_idx], # x-axis: mean radius
    X[:, y_idx], # y-axis: mean texture
    c=colors[y], # fill color = true class
    edgecolors=colors[best_pred], # border color = predicted class
    s=30,
    linewidths=0.6,
)
plt.xlabel(wdbc.feature_names[x_idx])
plt.ylabel(wdbc.feature_names[y_idx])
plt.title(
    f"WDBC Scatter Plot - {best_name}\n"
    f"fill = true class  |  border = predicted class"
)
plt.legend(handles=legend_handles, framealpha=0.6)
plt.tight_layout()
plt.savefig("wdbc_classification_scatter.png", dpi=150)
plt.close()
print("saved to 'wdbc_classification_scatter.png'")
