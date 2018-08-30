from sklearn import tree
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
#labels = ["apple", "apple", "orange",  "orange"]

#0 = apple and 1 = orange
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
# fit = find pattens and data
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))