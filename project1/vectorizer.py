import numpy as np
import pdb

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False
        self.feature = np.array([])

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)


        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            return (float(x) - mean) / std 

        return vectorizer

    def get_histogram_vectorizer(self, values):
        raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """

        unique_vals = np.unique(values)
        val_to_index = {val: i for i, val in enumerate(unique_vals)}

        def vectorizer(x):
            vector = [0] * len(unique_vals)
            vector[val_to_index[x]] = 1
            return vector
        
        return vectorizer


    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        self.is_fit = True

        for feature_type, features in self.feature_config.items():
            for feature in features:
                if feature_type == "numerical":
                    vals = np.array([float(x[feature]) for x in X])
                    self.feature_transforms[feature] = self.get_numerical_vectorizer(vals)
                elif feature_type == "categorical":
                    vals = np.array([x[feature] for x in X])
                    self.feature_transforms[feature] = self.get_categorical_vectorizer(vals)
        


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )
        
        transformed_data = []
        
        for x in X:
            transformed_features = []
            for feature, transform in self.feature_transforms.items():
                transformed_val = transform(x[feature])
                if isinstance(transformed_val, list):
                    transformed_features.extend(transformed_val)
                else:
                    transformed_features.append(transformed_val)
            transformed_data.append(transformed_features)
            
        return np.array(transformed_data)