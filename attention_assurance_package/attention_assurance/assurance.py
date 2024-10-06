from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    AutoTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
)
import torch
import requests
from io import BytesIO
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from attention_assurance.utils import FeatureExtractor, ZonesExtractor
from captum.attr import LayerGradientXActivation
from sklearn.ensemble import RandomForestClassifier


class Assurance:
    def __init__(
        self,
        sample_training_dataset: list,
        sample_training_dataset_labels: list[int],
        input_layer_name: str,
        number_of_layers: int,
        attentions_shape_per_layer: tuple,
        number_of_zones: int,  # TODO: best value can be found by elbow algorithim.
    ) -> None:
        self.__input_layer_name = input_layer_name
        self.__number_of_layers = number_of_layers
        self.__attentions_shape_per_layer = attentions_shape_per_layer
        self.__sample_dataset = sample_training_dataset
        self.__sample_labels = sample_training_dataset_labels
        self.__number_of_zones = number_of_zones
        self.__model = None

    @abstractmethod
    def __load_model_and_processor(self) -> tuple:
        pass

    @abstractmethod
    def __get_model_explainer(self) -> LayerGradientXActivation:
        pass

    @abstractmethod
    def __get_first_layer_for_gradient(self, inputs):
        pass

    # Method to calculate Transformer Attention Behavior Space (TABS)
    def __calculate_tabs(self, sample_dataset: list) -> tuple[np.ndarray, np.ndarray]:
        tabs = []
        predicted_labels = []
        for one_data_input in sample_dataset:
            inputs = self.processor(one_data_input, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()

            # Save predicted labels
            predicted_labels.append(predicted_label)

            # Calculate attributions
            attribution = self.__explainer.attribute(
                self.__get_first_layer_for_gradient(inputs), target=predicted_label
            )
            attribution = np.array(
                [
                    attribution[i][1]
                    .detach()
                    .numpy()
                    .reshape(self.__attentions_shape_per_layer)
                    for i in range(self.__number_of_layers)
                ]
            )

            # Get the attention weights
            all_attentions = outputs.attentions
            all_attentions_np = np.array(
                [
                    attention.detach()
                    .numpy()
                    .reshape(self.__attentions_shape_per_layer)
                    for attention in all_attentions
                ]
            )

            # Feature extraction
            feature_extractor = FeatureExtractor(
                attentions=all_attentions_np,
                attributions=attribution,
            )

            attentions_entropy = feature_extractor.get_attentions_entropy()
            attentions_positive_impact, attentions_negative_impact = (
                feature_extractor.get_attentions_impacts()
            )
            max_attention_value, mean_max_attention_value = (
                feature_extractor.get_attentions_confidences()
            )
            upper_triangle_jsd_matrix = feature_extractor.get_attentions_flow_change(
                return_matrix=False
            )
            attentions_sparsity = feature_extractor.get_attentions_sparsity()
            attention_class_weights_q = list(
                feature_extractor.get_attentions_distribution_on_classes()
            )
            attention_patches_weights_q = list(
                feature_extractor.get_attentions_distribution_on_patches()
            )
            attention_balance = feature_extractor.get_attentions_balance()
            attention_uniformity = feature_extractor.get_attentions_uniformity()

            # Adding New Features
            new_features = np.concatenate(
                (
                    attentions_entropy,
                    attentions_positive_impact,
                    attentions_negative_impact,
                    max_attention_value,
                    mean_max_attention_value,
                    upper_triangle_jsd_matrix,
                    attentions_sparsity,
                    attention_class_weights_q[0],
                    attention_class_weights_q[1],
                    attention_class_weights_q[2],
                    attention_class_weights_q[3],
                    attention_class_weights_q[4],
                    attention_patches_weights_q[0],
                    attention_patches_weights_q[1],
                    attention_patches_weights_q[2],
                    attention_patches_weights_q[3],
                    attention_patches_weights_q[4],
                    attention_balance,
                    attention_uniformity,
                ),
                axis=0,
            )

            tabs.append(new_features)

        tabs = np.array(tabs)
        predicted_labels = np.array(predicted_labels)
        return tabs, predicted_labels

    # Method to reduce the dimensions of the training TABS
    def __reduce_training_tabs_dimensions(
        self, tabs: np.ndarray, predicted_wrong_labels: np.ndarray
    ) -> tuple:
        clf = RandomForestClassifier()
        clf.fit(tabs, predicted_wrong_labels)

        feature_importance_threshold = 0.00065
        self.selected_feature = clf.feature_importances_ > feature_importance_threshold
        tabs = tabs[:, self.selected_feature]
        return tabs

    # Method to reduce the dimensions of the TABS after training
    def __reduce_after_training_tabs_dimensions(self, tabs: np.ndarray) -> tuple:
        tabs = tabs[:, self.selected_feature]
        return tabs

    # Method to normalize the training TABS
    def __normalize_training_tabs(self, tabs: np.ndarray):
        q1 = np.quantile(tabs, q=0.25, axis=0)
        q3 = np.quantile(tabs, q=0.75, axis=0)
        self.iqr = q3 - q1
        self.should_remove = self.iqr == 0
        self.iqr[self.should_remove] = 1
        self.median = np.median(tabs, axis=0)

        tabs = (tabs - self.median) / self.iqr
        tabs = tabs[:, np.logical_not(self.should_remove)]
        return tabs

    # Method to normalize the TABS after training
    def __normalize_after_training_tabs(self, tabs: np.ndarray):
        tabs = (tabs - self.median) / self.iqr
        tabs = tabs[:, np.logical_not(self.should_remove)]
        return tabs

    # Method to count training labels per zone
    def __count_training_labels_per_zone(
        self, tabs: np.ndarray, predicted_labels: list
    ):
        zones = self.zone_extractor.find_nearest_zone(tabs)
        labels_per_zone_df = pd.DataFrame()
        labels_per_zone_df["zones"] = zones
        labels_per_zone_df["label"] = predicted_labels
        self.__labels_per_zone_df = labels_per_zone_df.drop_duplicates(
            ["zones", "label"]
        ).reset_index(drop=True)

    # Method to create behavior space
    def create_behavior_space(self):
        self.__model, self.processor = self.__load_model_and_processor()
        self.__explainer = self.__get_model_explainer()
        tabs, predicted_labels = self.__calculate_tabs(self.__sample_dataset)
        predicted_wrong_labels = np.array(
            predicted_labels != self.__sample_labels
        ).astype(int)
        tabs = self.__reduce_training_tabs_dimensions(tabs, predicted_wrong_labels)
        tabs = self.__normalize_training_tabs(tabs)
        self.zone_extractor = ZonesExtractor(self.__number_of_zones)
        self.zone_extractor.fit(tabs)
        self.__count_training_labels_per_zone(self, tabs, predicted_labels)
        self.online_error_predictor = RandomForestClassifier(max_depth=None)
        self.online_error_predictor.fit(tabs, predicted_wrong_labels)

    # Method to calculate input dataset TABS
    def __calculate_input_dataset_tabs(self, input_dataset: list) -> tuple:
        tabs, predicted_labels = self.__calculate_tabs(input_dataset)
        tabs = self.__reduce_after_training_tabs_dimensions(tabs)
        tabs = self.__normalize_after_training_tabs(tabs)
        return tabs, predicted_labels

    # Method to evaluate test dataset coverage
    def evaluate_test_dataset_coverage(
        self, test_dataset: list, test_dataset_labels: list
    ):
        tabs, predicted_labels = self.__calculate_input_dataset_tabs(test_dataset)
        zones = self.zone_extractor.find_nearest_zone(tabs)

        test_labels_per_zone_df = pd.DataFrame()
        test_labels_per_zone_df["zones"] = zones
        test_labels_per_zone_df["label"] = predicted_labels
        test_labels_per_zone_df = test_labels_per_zone_df.drop_duplicates(
            ["zones", "label"]
        ).reset_index(drop=True)

        # Calculate BCC
        common_labels_df = pd.merge(
            self.__labels_per_zone_df,
            test_labels_per_zone_df,
            how="inner",
            on=["zones", "label"],
        )
        BCC = len(common_labels_df) / len(self.__labels_per_zone_df)

        # Calculate NCC
        all_classes = set(self.__sample_labels)
        new_labels_df = pd.merge(
            self.__labels_per_zone_df,
            test_labels_per_zone_df,
            how="right",
            on=["zones", "label"],
        )
        new_labels_df = new_labels_df[new_labels_df.isnull().any(axis=1)]
        NCC = len(new_labels_df) / (len(all_classes) - len(self.__labels_per_zone_df))

        # Calculate ECC
        extended_labels_df = pd.merge(
            self.__labels_per_zone_df,
            test_labels_per_zone_df,
            how="outer",
            on=["zones", "label"],
        )
        extended_labels_df = extended_labels_df[extended_labels_df.isnull().any(axis=1)]
        ECC = len(extended_labels_df) / (len(zones) * len(all_classes))

        return BCC, NCC, ECC

    # Method to predict error probability
    def predit_error_probability(self, input_dataset: list):
        tabs, _ = self.__calculate_input_dataset_tabs(input_dataset)
        y_p = self.online_error_predictor.predict_proba(tabs)[:, 1]
        return y_p
