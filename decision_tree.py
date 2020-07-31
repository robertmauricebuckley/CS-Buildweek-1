import numpy as np

class Node:  #
    def __init__(self, gini, samples, samples_per_class, predicted_class):
        self.gini = gini
        self.samples = samples
        self.samples_per_class = samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.split = 0
        self.left = None
        self.right = None


class Decision_Tree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def _gini(self, y):

        outcomes = y.size
        return 1.0 - sum((np.sum(y == c) / outcomes) ** 2 for c in range(self.num_outcomes))

    def find_split(self, X, y):
        """
        find the feature to use for the next node split
        and also find where the plit should be in that feature
        This loops through the split options within a feature to find the best gini score,
        then it loops through each feature to compare optimal gini scores
        """
        choices = y.size
        if choices <= 1:
            return None, None

        # find the number of each option in the current node.
        options_parent = [np.sum(y == c) for c in range(self.num_outcomes)]

        # find the gini of current node.
        best_gini = 1.0 - sum((n / choices) ** 2 for n in options_parent)
        best_idx, best_split = None, None

        # loop through the features to get splits and options.
        for idx in range(self.num_features):
            splits, options = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.num_outcomes
            num_right = options_parent.copy()
            for i in range(1, choices):
                c = options[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.num_outcomes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / i) ** 2 for x in range(self.num_outcomes)
                )

                gini = (i * gini_left + (choices - i) * gini_right) / choices

                if splits[i] == splits[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_split = (splits[i] + splits[i - 1]) / 2

        return best_idx, best_split

    def fit(self, X, y):
        self.num_outcomes = len(set(y))
        self.num_features = X.shape[1]
        self.tree_ = self.extd_tree(X, y)

    def extd_tree(self, X, y, depth=0):
        '''
        adds nodes to the tree and splits the data by using the find_split method
        Continues to grow tree until maximum depth has been reached or
        until current nodes gini score is higher than remaining split options.
        '''
        
        samples_per_class = [np.sum(y == i) for i in range(self.num_outcomes)]
        predicted_class = np.argmax(samples_per_class)
        node = Node(
            gini=self._gini(y),
            samples=y.size,
            samples_per_class=samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, splt = self.find_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < splt
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.split = splt
                node.left = self.extd_tree(X_left, y_left, depth + 1)
                node.right = self.extd_tree(X_right, y_right, depth + 1)
        return node


    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.split:
                node = node.left
            else:
                node = node.right
        return node.predicted_class



