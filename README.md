# from-attention-to-assurance
### From Attention to Assurance: Enhancing Transformer Encoder Reliability through Advanced Testing and Online Error Prediction

The Transformer architecture has revolutionized the fields of natural language processing and computer vision by leveraging self-attention mechanisms for enhanced model performance and parallel computation capabilities. Despite their success, ensuring the comprehensiveness of testing and the reliability of Transformer models, especially in critical applications, remains a challenge. This study introduces a novel framework for evaluating the test suites of Transformer models, specifically focusing on the encoder architecture. We define a comprehensive behavior space for Transformer attentions, characterized by features such as attention confidence, impact, focus, and distribution attributes. Based on this behavior space, we propose three innovative coverage criteria: New Centroid Coverage, Basic Centroid Coverage, and Extended Centroid Coverage, designed to ensure thorough testing by targeting diverse behavioral zones within the models. To enhance the reliabilty of the model, we develop an online error prediction model using the defined behavior space, achieving high predictive accuracy, as evidenced by area under the curve (AUC) values of approximately 84\% for the Vision Transformer (ViT) and 93\% for BERT.

This repository contains the code and data required to replicate the experiments presented in the research paper "From Attention to Assurance: Enhancing Transformer Encoder Reliability through Advanced Testing and Online Error Prediction". 
Banking77 dataset is available in the repository. Also, link to the image dataset is available in the codes.

## Citation

Please cite our paper if you use our method, code, or data in your work:

https://

## License

This repository is licensed under the [MIT License](LICENSE).
