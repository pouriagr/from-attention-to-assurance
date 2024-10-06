from captum.attr import LayerGradientXActivation
from assurance import Assurance
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
from captum.attr import LayerGradientXActivation


class BERTAssurance(Assurance):
    def __init__(
        self,
        sample_training_dataset: list,
        sample_training_dataset_labels: list[int],
        number_of_zones: int,
    ) -> None:
        super().__init__(
            sample_training_dataset=sample_training_dataset,
            sample_training_dataset_labels=sample_training_dataset_labels,
            input_layer_name="",
            number_of_layers=12,
            attentions_shape_per_layer=(12, 197, 197),
            number_of_zones=number_of_zones,
        )
        (inputs["input_ids"], inputs["attention_mask"])

    def __load_model_and_processor(self):
        model_id = "philschmid/BERT-Banking77"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model = model.eval()
        model.config.output_attentions = True
        processor = AutoTokenizer.from_pretrained(model_id)
        return model, processor

    def __get_model_explainer(self) -> LayerGradientXActivation:
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, input_ids, attention_mask=None):
                embeddings = self.model.bert.embeddings(input_ids)
                embeddings_leaf = embeddings.detach().clone().requires_grad_(True)
                self.outputs = self.model(
                    inputs_embeds=embeddings_leaf, attention_mask=attention_mask
                )
                return self.outputs.logits

        wrapper_model = ModelWrapper(self.__model)
        # Create Guided Backprop explainer
        layers = [
            wrapper_model.model.bert.encoder.layer[i].attention.self
            for i in range(self.__number_of_layers)
        ]
        explainer = LayerGradientXActivation(wrapper_model, layers)
        return explainer

    def __get_first_layer_for_gradient(self, inputs):
        return (inputs["input_ids"], inputs["attention_mask"])
