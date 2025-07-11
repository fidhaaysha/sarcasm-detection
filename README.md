BERT Sarcasm Detection in News Headlines



A fine-tuned BERT model for detecting sarcasm in news headlines, achieving 85.67% accuracy on test data. This project demonstrates the power of transfer learning with pre-trained language models for nuanced NLP tasks.



 Results

Metric	Value
Test Accuracy	85.67%
Training Loss	0.325
Evaluation Loss	0.609
Training Time	~12 minutes
 Features

Fine-tuned BERT Model: Pre-trained bert-base-uncased adapted for sarcasm detection
Interactive Interface: Real-time prediction with confidence scores
Comprehensive Analysis: Detailed performance metrics and visualizations
Ready-to-Deploy: Saved model artifacts for immediate use
Well-Documented: Extensive code comments and methodology explanation
Model Performance

The model successfully distinguishes between sarcastic and non-sarcastic headlines:

# Example predictions
predict("Local Man Discovers Incredible Weight Loss Secret: Eating Less")
# Output: Sarcastic (92.3% confidence)

predict("Scientists Develop New Cancer Treatment")
# Output: Not Sarcastic (88.7% confidence)


 Dataset

The project uses the Sarcasm Headlines Dataset containing:

26,709 headlines from The Onion and HuffPost
Binary labels: Sarcastic (1) vs Non-sarcastic (0)
Balanced distribution for training stability
Subset of 3,000 samples used for efficient training
Data Source

The Onion: Satirical news headlines (sarcastic)
HuffPost: Legitimate news headlines (non-sarcastic)
Model Architecture

Base Model

BERT-base-uncased: 110M parameters
12-layer transformer encoder
WordPiece tokenization: 30,000 vocabulary
Fine-tuning Setup

Classification head: Additional linear layer for binary classification
Max sequence length: 128 tokens
Optimizer: AdamW with default learning rate (5e-5)
Training epochs: 2 (efficient training)
Training Details

Hyperparameters

Batch size: 8 (per device)
Learning rate: 5e-5 (AdamW default)
Epochs: 2
Max sequence length: 128 tokens
Warmup steps: Default
Data Split

Training: 80% (2,400 samples)
Validation: 10% (300 samples)
Test: 10% (300 samples)
Training Environment

Framework: PyTorch + Hugging Face Transformers
Hardware: GPU-accelerated training (recommended)
Time: ~12 minutes for 2 epochs
Evaluation Metrics

Test Set Performance

Accuracy: 85.67%


Model Efficiency
Training samples/second: 6.705
Evaluation samples/second: 32.695
Model size: ~440MB (BERT-base + classification head)
License

This project is licensed under the MIT License - https://creativecommons.org/licenses/by/4.0/



References

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Misra, R., & Arora, P. (2019). Sarcasm Detection using News Headlines Dataset. AI Open.
Wolf, T., et al. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing.
