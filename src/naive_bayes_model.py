import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        self.unique_classes = None
        self.priors = None
        self.means = None
        self.variances = None
        
    def fit(self, X, y):
        """Trains model on data. Calculates means, variances, and priors for each class"""
        self.priors = self.calculate_prior_probs(y)
        self.unique_classes = np.unique(y)
        num_classes = len(self.unique_classes)
        num_features = X.shape[1]

        self.means = np.zeros((num_classes, num_features))
        self.variances = np.zeros((num_classes, num_features))
        for idx, c in enumerate(self.unique_classes):
            X_c = X[y == c]  # All reviews of class c
            self.means[idx] = X_c.mean(axis=0) # Calculates the mean of the column
            self.variances[idx] = X_c.var(axis=0)
        
        # Add small epsilon to variances to prevent division by zero
        self.variances = np.maximum(self.variances, 1e-10)            

    def calculate_likelihood(self, x, mean, var):
        """Calculates the Gaussian likelihood of the data with the given mean and variance"""
        return  -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
        

    def predict(self, X):
        """Performs inference using Bayes' Theorem: P(A | B) = P(B | A) * P(A) / P(B)
        We have to add the log-likelihoods given each feature"""
        predictions = []
        # For each sample, we have to take each feature(word) in the sample and get the likelihood
        for x in X:
            class_scores = []
            # we must do this for each class, then find the highest score for each sample and classify
            for class_idx, c in enumerate(self.unique_classes):
                prior = np.log(self.priors[class_idx])
                log_likelihood = np.sum(self.calculate_likelihood(x, self.means[class_idx], self.variances[class_idx]))
                posterior = prior + log_likelihood
                class_scores.append(posterior)
            predicted_class = np.argmax(class_scores)
            predictions.append(self.unique_classes[predicted_class])
        return predictions

    def calculate_prior_probs(self, y):
        """Calculate proportion of words taken up by each class"""
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        num_classes = len(unique)
        prior_probs = np.zeros(num_classes)
        for idx, c in enumerate(unique):
            prior_probs[idx] = counts[idx] / total_samples
        return prior_probs