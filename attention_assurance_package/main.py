from attention_assurance import BERTAssurance, ViTAssurance

def main():
    # Define sample data for text and image models
    text_sample_data = ["hello", "how are you?"]  # Sample dataset
    text_sample_data_labels = [0, 1]  # Corresponding labels

    image_sample_data = ["image_obj1", "image_obj2"]  # Sample dataset
    image_sample_data_labels = [0, 1]  # Corresponding labels

    # Define the number of zones for each model
    num_zones = 10  # This can be determined using the Elbow method

    # Create instances of BERT and ViT assurance classes
    bert_assurance = BERTAssurance(text_sample_data, text_sample_data_labels, num_zones)
    vit_assurance = ViTAssurance(image_sample_data, image_sample_data_labels, num_zones)

    # Call the create_behavior_space method for each instance to generate the behavior space
    bert_assurance.create_behavior_space()
    vit_assurance.create_behavior_space()

    # Define test data for text and image models
    test_text_sample_data = ["hello", "how are you?"]  # Test dataset
    test_text_sample_data_labels = [0, 1]  # Corresponding labels

    test_image_sample_data = ["image_obj1", "image_obj2"]  # Test dataset
    test_image_sample_data_labels = [0, 1]  # Corresponding labels

    # Evaluate test dataset coverage for each instance
    BCC_bert, NCC_bert, ECC_bert = bert_assurance.evaluate_test_dataset_coverage(test_text_sample_data, test_text_sample_data_labels)
    BCC_vit, NCC_vit, ECC_vit = vit_assurance.evaluate_test_dataset_coverage(test_image_sample_data, test_image_sample_data_labels)

    # Print the results
    print(f'BERT model - BCC: {BCC_bert}, NCC: {NCC_bert}, ECC: {ECC_bert}')
    print(f'ViT model - BCC: {BCC_vit}, NCC: {NCC_vit}, ECC: {ECC_vit}')

    # Predict error probability for each instance
    error_prob_bert = bert_assurance.predit_error_probability(test_text_sample_data)
    error_prob_vit = vit_assurance.predit_error_probability(test_image_sample_data)

    # Print the results
    print(f'BERT model - Error Probability: {error_prob_bert}')
    print(f'ViT model - Error Probability: {error_prob_vit}')

if __name__ == "__main__":
    main()
    
# Note: To extend this framework to other Transformer models, you can create new assurance classes by inheriting from the base 'Assurance' class 
# and implementing the required methods. This will allow you to evaluate and predict the performance of any Transformer model.