# Enhanced Movie Dialogue Chatbot

A sophisticated neural conversational AI trained on movie dialogues, featuring advanced attention mechanisms, beam search decoding, and contextual conversation handling.

## Overview

This project implements a sequence-to-sequence (Seq2Seq) neural network chatbot that learns conversational patterns from movie dialogue datasets. The model uses encoder-decoder architecture with Luong attention mechanism and beam search for generating more coherent and contextually appropriate responses.

## Features

### Core Architecture
- **Bidirectional GRU Encoder**: Captures context from both directions in input sequences
- **Attention-based Decoder**: Uses Luong attention mechanism for better context awareness
- **Beam Search Decoding**: Generates multiple response candidates and selects the best one
- **Advanced Preprocessing**: Handles Unicode normalization, text cleaning, and vocabulary management

### Training Enhancements
- **Scheduled Sampling**: Gradually reduces teacher forcing during training
- **Learning Rate Scheduling**: Adaptive learning rate decay for better convergence
- **Gradient Clipping**: Prevents exploding gradients during training
- **Dropout Regularization**: Reduces overfitting with configurable dropout rates

### Interactive Features
- **Enhanced Chat Interface**: User-friendly conversation system with command support
- **Conversation History**: Maintains context across multiple exchanges
- **Sentiment Analysis**: Optional NLTK-based sentiment awareness
- **Response Quality Control**: Fallback responses for better user experience

## Requirements

```python
torch>=1.9.0
numpy
nltk
unicodedata
json
csv
codecs
itertools
math
time
datetime
random
re
os
```

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install torch numpy nltk
```
3. Download NLTK resources (optional, for sentiment analysis):
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## Dataset Requirements

The chatbot expects a dataset in JSONL format with the following structure:
```json
{
  "id": "line_id",
  "speaker": "character_name", 
  "text": "dialogue_text",
  "conversation_id": "conv_id",
  "meta": {"movie_id": "movie_identifier"}
}
```

Compatible with datasets like:
- Cornell Movie-Dialogs Corpus
- Custom movie dialogue datasets in similar format

## Usage

### Basic Setup and Training

```python
# Run the chatbot with training
python chatbot.py

# Choose option 1 to train a new model
# Or option 2 to load an existing checkpoint
```

### Configuration Parameters

```python
# Model Configuration
hidden_size = 768          # Hidden layer dimensions
encoder_n_layers = 3       # Number of encoder layers
decoder_n_layers = 3       # Number of decoder layers
dropout = 0.2              # Dropout rate
batch_size = 64            # Training batch size
attn_model = 'general'     # Attention mechanism type

# Training Parameters
learning_rate = 0.0001     # Base learning rate
n_epochs = 50              # Number of training epochs
teacher_forcing_ratio = 0.7 # Initial teacher forcing ratio
clip = 50.0                # Gradient clipping threshold
```

### Loading Pre-trained Models

```python
# Load existing checkpoint
run_enhanced_chatbot(load_checkpoint="path/to/checkpoint.tar")
```

## Architecture Details

### Encoder (EncoderRNN)
- Bidirectional GRU for processing input sequences
- Embedding layer for word representations
- Packing/unpacking for efficient batch processing
- Linear layer to combine bidirectional outputs

### Attention Mechanism (Attn)
Multiple attention types supported:
- **Dot-product**: Simple dot product attention
- **General**: Learnable linear transformation
- **Concat**: Concatenation-based attention
- **Scaled Dot-product**: Transformer-style attention

### Decoder (LuongAttnDecoderRNN)
- Unidirectional GRU with attention integration
- Context vector computation using attention weights
- Layer normalization for training stability
- Softmax output layer for token prediction

### Beam Search Decoder
- Maintains multiple hypothesis during decoding
- Configurable beam size (default: 5)
- Length-normalized scoring
- Early stopping on EOS tokens

## Data Processing Pipeline

1. **Raw Data Loading**: Parse JSONL movie dialogue files
2. **Conversation Extraction**: Group lines into dialogue pairs
3. **Text Normalization**: Unicode handling, lowercasing, punctuation
4. **Vocabulary Building**: Word-to-index mapping with special tokens
5. **Filtering**: Remove overly long sequences and rare words
6. **Batch Processing**: Create training batches with padding

## Training Process

### Multi-phase Training
1. **Data Preparation**: Load and preprocess dialogue pairs
2. **Model Initialization**: Set up encoder-decoder with embeddings
3. **Iterative Training**: Batch-wise gradient descent with attention
4. **Checkpoint Saving**: Regular model state preservation
5. **Evaluation**: Performance metrics and response quality assessment

### Advanced Training Features
- **Scheduled Sampling**: Reduces teacher forcing over time
- **Learning Rate Decay**: Exponential decay for convergence
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Based on validation performance

## Interactive Chat Interface

### Available Commands
- `quit` / `exit` - End conversation
- `clear` - Clear chat history
- `history` - Show recent exchanges
- `save` - Export conversation to file
- `help` - Display command list

### Response Generation
1. Input normalization and tokenization
2. Encoder processing of input sequence
3. Beam search decoding with attention
4. Post-processing and quality filtering
5. Contextual fallback if needed

## Model Evaluation

### Automated Metrics
- **BLEU Score**: Translation quality measurement
- **Response Length**: Average words per response
- **Lexical Diversity**: Vocabulary richness
- **Contextual Appropriateness**: Manual evaluation scoring

### Performance Analysis
```python
# Evaluate model performance
evaluate_chatbot_performance(encoder, decoder, searcher, vocab, test_pairs)
```

## File Structure

```
project/
├── data/
│   ├── movie-corpus/
│   │   ├── utterances.jsonl           # Input dataset
│   │   └── formatted_movie_lines.txt  # Processed pairs
│   └── save/
│       └── model_checkpoints/         # Saved models
├── chatbot.py                         # Main implementation
└── README.md                          # This file
```

## Customization Options

### Model Architecture
- Adjust hidden dimensions for complexity/speed tradeoff
- Modify number of layers for deeper representations
- Change attention mechanism type
- Configure dropout rates for regularization

### Training Parameters
- Batch size based on available memory
- Learning rate scheduling strategies
- Teacher forcing ratio schedules
- Gradient clipping thresholds

### Response Generation
- Beam search beam size
- Maximum response length
- Temperature sampling
- Fallback response strategies

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size or hidden_size
2. **Poor Responses**: Increase training epochs or data size
3. **Slow Training**: Use GPU acceleration or reduce model complexity
4. **Repetitive Outputs**: Adjust beam search parameters or add diversity penalty

### Performance Optimization
- Use CUDA for GPU acceleration
- Implement gradient accumulation for large batches
- Add learning rate warmup
- Use mixed precision training

## Future Enhancements

### Potential Improvements
- **Transformer Architecture**: Upgrade to transformer-based models
- **Multi-turn Context**: Better conversation history handling
- **Personality Modeling**: Character-specific response styles
- **Emotion Recognition**: Sentiment-aware response generation
- **Knowledge Integration**: External knowledge base incorporation

### Advanced Features
- **Fine-tuning**: Domain-specific adaptation
- **Reinforcement Learning**: Human feedback optimization
- **Multi-modal Input**: Image and text understanding
- **Real-time Learning**: Continuous model updates

## License

This project is intended for educational and research purposes. Please ensure compliance with dataset licensing terms when using movie dialogue corpora.

## Contributing

Contributions are welcome! Areas for improvement:
- Better attention mechanisms
- More sophisticated beam search
- Enhanced conversation context handling
- Performance optimizations
- Additional evaluation metrics

## Citation

If you use this code in your research, please cite:
- Luong et al. "Effective Approaches to Attention-based Neural Machine Translation"
- Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate"
