from scipy.stats import entropy
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.linalg import sqrtm
import numpy as np


class FeatureExtractor:
    def __init__(self, attentions: np.array, attributions: np.array):
        self.attentions = attentions
        self.attributions = attributions
        self.total_tokens = attentions.shape[3]

    def get_attentions_entropy(self) -> np.array:
        return np.apply_along_axis(
            entropy,
            1,
            self.attentions.reshape(-1, self.total_tokens * self.total_tokens),
        )

    def get_attentions_impacts(self) -> tuple[np.array, np.array]:
        def positive_mask(array):
            array[array < 0] = 0
            return array

        def negative_mask(array):
            array[array > 0] = 0
            return array

        attentions_positive_impact = (
            positive_mask(self.attributions.reshape(12, 12, -1).copy())
            .sum(axis=2)
            .flatten()
        )
        attentions_negative_impact = (
            negative_mask(self.attributions.reshape(12, 12, -1).copy())
            .sum(axis=2)
            .flatten()
        )
        return attentions_positive_impact, attentions_negative_impact

    def get_attentions_confidences(self) -> tuple[np.array, np.array]:
        max_attention_value = self.attentions.reshape(
            -1, self.total_tokens * self.total_tokens
        ).max(axis=1)
        mean_max_attention_value = (
            self.attentions.reshape(-1, self.total_tokens, self.total_tokens)
            .max(axis=2)
            .mean(axis=1)
        )
        return max_attention_value, mean_max_attention_value

    def get_attentions_flow_change(self, return_matrix=True) -> np.array:
        def jensen_shannon(p, q):
            """
            method to compute the Jenson-Shannon Distance
            between two probability distributions
            """
            # convert the vectors into numpy arrays in case they aren't
            p = np.array(p)
            q = np.array(q)

            # calculate m
            m = 0.5 * (p + q)

            # compute Jensen Shannon Divergence
            return 0.5 * (entropy(p, m) + entropy(q, m))

        # reshape your data
        all_attentions_mean = self.attentions.reshape(
            12, 12, self.total_tokens * self.total_tokens
        ).mean(axis=1)

        # calculate the Jensen Shannon divergence matrix
        jsd_vector = distance.pdist(all_attentions_mean, metric=jensen_shannon)

        # convert the condensed distance vector to a square distance matrix
        jsd_matrix = distance.squareform(jsd_vector)
        if return_matrix:
            return jsd_matrix
        else:
            return jsd_matrix[np.triu_indices_from(jsd_matrix, 1)].flatten()

    def get_attentions_sparsity(self) -> np.array:
        values = self.attentions.reshape(12, 12, -1)
        threshold = np.min(values, axis=2).max()
        attentions_sparsity = (
            np.sum(values <= threshold, axis=2) / values.shape[2]
        ).reshape(-1)
        return attentions_sparsity

    def get_attentions_distribution_on_classes(
        self,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        attention_class_weights = self.attentions[:, :, 0, 1:]
        attention_class_weights_q2 = np.quantile(
            attention_class_weights, q=0.50, axis=2
        ).reshape(-1)
        attention_class_weights_q0 = (
            np.quantile(attention_class_weights, q=0, axis=2).reshape(-1)
            - attention_class_weights_q2
        )
        attention_class_weights_q1 = (
            np.quantile(attention_class_weights, q=0.25, axis=2).reshape(-1)
            - attention_class_weights_q2
        )
        attention_class_weights_q3 = (
            np.quantile(attention_class_weights, q=0.75, axis=2).reshape(-1)
            - attention_class_weights_q2
        )
        attention_class_weights_q4 = (
            np.quantile(attention_class_weights, q=1, axis=2).reshape(-1)
            - attention_class_weights_q2
        )
        return (
            attention_class_weights_q0,
            attention_class_weights_q1,
            attention_class_weights_q2,
            attention_class_weights_q3,
            attention_class_weights_q4,
        )

    def get_attentions_distribution_on_patches(
        self,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        attention_patches_weights = self.attentions[:, :, 1:, 1:].reshape(12, 12, -1)
        attention_patches_weights_q2 = np.quantile(
            attention_patches_weights, q=0.50, axis=2
        ).reshape(-1)
        attention_patches_weights_q0 = (
            np.quantile(attention_patches_weights, q=0, axis=2).reshape(-1)
            - attention_patches_weights_q2
        )
        attention_patches_weights_q1 = (
            np.quantile(attention_patches_weights, q=0.25, axis=2).reshape(-1)
            - attention_patches_weights_q2
        )
        attention_patches_weights_q3 = (
            np.quantile(attention_patches_weights, q=0.75, axis=2).reshape(-1)
            - attention_patches_weights_q2
        )
        attention_patches_weights_q4 = (
            np.quantile(attention_patches_weights, q=1, axis=2).reshape(-1)
            - attention_patches_weights_q2
        )
        return (
            attention_patches_weights_q0,
            attention_patches_weights_q1,
            attention_patches_weights_q2,
            attention_patches_weights_q3,
            attention_patches_weights_q4,
        )

    def get_attentions_balance(self) -> np.array:
        class_attention_vector = self.attentions[:, :, 0, 1:]
        patches_attention_vector = self.attentions[:, :, 1:, 1:].sum(axis=2)

        p = class_attention_vector
        q = patches_attention_vector

        attention_balance = ((p - q) ** 2).sum(axis=2) ** 0.5

        return attention_balance.reshape(-1)

    def get_attentions_uniformity(self) -> np.array:
        attention_uniformity = np.std(
            self.attentions.reshape(-1, self.total_tokens * self.total_tokens), axis=1
        )
        return attention_uniformity
