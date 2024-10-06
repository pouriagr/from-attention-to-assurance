# from-attention-to-assurance
### From Attention to Assurance: Enhancing Transformer Encoder Reliability through Advanced Testing and Online Error Prediction

The Transformer architecture has revolutionized the fields of natural language processing and computer vision by leveraging self-attention mechanisms for enhanced model performance and parallel computation capabilities. Despite their success, ensuring the comprehensiveness of testing and the reliability of Transformer models, especially in critical applications, remains a challenge. This study introduces a novel framework for evaluating the test suites of Transformer models, specifically focusing on the encoder architecture. We define a comprehensive behavior space for Transformer attentions, characterized by features such as attention confidence, impact, focus, and distribution attributes. Based on this behavior space, we propose three innovative coverage criteria: New Centroid Coverage, Basic Centroid Coverage, and Extended Centroid Coverage, designed to ensure thorough testing by targeting diverse behavioral zones within the models. To enhance the reliability of the model, we develop an online error prediction model using the defined behavior space, achieving high predictive accuracy, as evidenced by area under the curve (AUC) values of approximately 84\% for the Vision Transformer (ViT) and 93\% for BERT.

This repository contains the code and data required to replicate the experiments presented in the research paper "From Attention to Assurance: Enhancing Transformer Encoder Reliability through Advanced Testing and Online Error Prediction". The code is organized into a Python package named `attention_assurance` which includes classes for BERT and ViT models. These classes inherit from a base `Assurance` class and implement methods for defining the behavior space, calculating coverage criteria, and predicting error probability. The `main.py` script demonstrates how to use these classes to evaluate test dataset coverage and predict error probability for BERT and ViT models. 

The Banking77 dataset used in the paper is available in the repository. Also, a link to the ImageNet dataset used for the ViT model is provided in the code.

## Installation

To install the `attention_assurance` package, clone this repository and run the following command in the repository directory:

```bash
pip install poetry
poetry install
```

## Usage

After installing the `attention_assurance` package, you can import the `BERTAssurance` and `ViTAssurance` classes in your Python script as follows:

```python
from attention_assurance import BERTAssurance, ViTAssurance
```

You can then create instances of these classes and call their methods to evaluate test dataset coverage and predict error probability. See the `main.py` script for a detailed example.

## Development
To extend this framework to other Transformer models, you can create new assurance classes by inheriting from the base `Assurance` class and implementing the required methods. 
This will allow you to evaluate and predict the performance of any Transformer model.

## Citation

Please cite our paper if you use our method, code, or data in your work:

https://

## License

This repository is licensed under the [MIT License](LICENSE).
