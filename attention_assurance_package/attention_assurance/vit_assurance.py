from assurance import Assurance
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)
import torch
from captum.attr import LayerGradientXActivation


class ViTAssurance(Assurance):
    def __init__(
        self,
        sample_training_dataset: list,
        sample_training_dataset_labels: list[int],
        number_of_zones: int,
    ) -> None:
        super().__init__(
            sample_training_dataset=sample_training_dataset,
            sample_training_dataset_labels=sample_training_dataset_labels,
            input_layer_name="pixel_values",
            number_of_layers=12,
            attentions_shape_per_layer=(12, 197, 197),
            number_of_zones=number_of_zones,
        )

    def __load_model_and_processor(self):
        model_id = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(model_id)
        model = model.eval()
        model.config.output_attentions = True
        processor = ViTImageProcessor.from_pretrained(model_id)
        return model, processor

    def __get_model_explainer(self) -> LayerGradientXActivation:
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, pixel_values):
                self.outputs = self.model(pixel_values)
                return self.outputs.logits

        wrapper_model = ModelWrapper(self.__model)
        # Create Guided Backprop explainer
        layers = [
            wrapper_model.model.vit.encoder.layer[i].attention.attention
            for i in range(self.__number_of_layers)
        ]
        explainer = LayerGradientXActivation(wrapper_model, layers)
        return explainer

    def __get_first_layer_for_gradient(self, inputs):
        inputs["pixel_values"].requires_grad = True
        return inputs["pixel_values"]
