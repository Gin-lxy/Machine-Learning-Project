# Deception Detection in Narrated Stories: A Deep Learning Approach

## Abstract

This report presents an automated approach for detecting deception in
narrated stories using deep learning techniques. We develop a novel
audio-based deception detection system that combines mel-spectrograms
and Mel-frequency cepstral coefficients (MFCCs) using a
transformer-based architecture. The system processes audio recordings of
3-5 minutes in duration and predicts whether the narrated story is true
or false.Code and project details are available at https://github.com/Gin-lxy/Machine-Learning-Project.git

## 1. Introduction

Deception detection in spoken narratives presents a challenging and
important problem in various fields, including security, psychology, and
human-computer interaction. Traditional approaches often rely on human
expertise and are subject to bias and inconsistency. This project
develops an automated system that can analyze audio recordings and
determine the veracity of narrated stories.

## 2. Methodology

### 2.1 Data Processing Pipeline

The system implements a comprehensive data processing pipeline
consisting of three main stages:

1.  **Audio Feature Extraction**:
    -   Mel-spectrogram generation (128 mel bands)
    -   MFCC extraction (20 coefficients)
    -   Length standardization to 130 time steps
2.  **Feature Transformation**:
    -   Linear embedding of mel-spectrograms and MFCCs
    -   Positional encoding for sequence information
    -   Feature fusion through concatenation
3.  **Classification**:
    -   Transformer-based sequence processing
    -   Global average pooling
    -   Multi-layer classification head

### 2.2 Model Architecture

The core of our system is the AudioTransformer model, which consists of
several key components:

``` python
class AudioTransformer(nn.Module):
    def __init__(self, mel_channels=128, mfcc_channels=20, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        # Feature embedding layers
        self.mel_embed = nn.Linear(mel_channels, d_model)
        self.mfcc_embed = nn.Linear(mfcc_channels, d_model)
        
        # Positional encoding for sequence information
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

The model architecture incorporates several innovative design choices:

1.  **Dual-stream Processing**: The model processes mel-spectrograms and
    MFCCs in parallel streams before fusion, allowing it to capture both
    spectral and cepstral information.

2.  **Positional Encoding**: A sinusoidal positional encoding scheme
    helps the model understand temporal relationships in the audio
    features:

``` python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

1.  **Multi-head Attention**: The transformer encoder uses 8 attention
    heads to capture different types of relationships in the audio
    features.

### 2.3 Training Strategy

The training process incorporates several techniques to improve model
robustness:

1.  **Learning Rate Management**:
    -   Initial learning rate: 0.0001
    -   Learning rate reduction on plateau
    -   Patience of 5 epochs for learning rate adjustment
2.  **Regularization**:
    -   Dropout (0.1) in transformer layers and classification head
    -   Gradient clipping with max norm 1.0
    -   Early stopping with patience of 10 epochs
3.  **Loss Function**:
    -   Binary Cross Entropy Loss for binary classification
    -   Model outputs probability through sigmoid activation

## 3. Implementation Details

The implementation includes several practical considerations for robust
deployment:

### 3.1 Audio Processing

``` python
def load_audio_features(file_path, fixed_length=130):
    # Load and resample audio
    y, sr = librosa.load(file_path)
    
    # Extract features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
```

### 3.2 Dataset Management

A custom PyTorch dataset class handles data loading and batching:

``` python
class DeceptionDataset(torch.utils.data.Dataset):
    def __init__(self, features_list, labels):
        self.features = features_list
        self.labels = labels
    
    def __getitem__(self, idx):
        mel_spec = torch.FloatTensor(self.features[idx]['mel_spec']).unsqueeze(0)
        mfcc = torch.FloatTensor(self.features[idx]['mfcc'])
        label = torch.FloatTensor([self.labels[idx]])
        return {'mel_spec': mel_spec, 'mfcc': mfcc, 'label': label}
```

### 3.3 Inference System

The StoryPredictor class provides a clean interface for making
predictions:

``` python
class StoryPredictor:
    def __init__(self, model_path='best_model.pth'):
        self.model = AudioTransformer()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, audio_path):
        features = self.process_audio(audio_path)
        with torch.no_grad():
            output = self.model(mel_spec, mfcc)
            probability = output.item()
            prediction = probability > 0.5
        return prediction, probability
```

## 4. Experimental Results and Analysis

### 4.1 Training Dynamics

The model was trained for 28 epochs before early stopping was triggered.
The training process revealed several key observations:

1.  **Loss Trends**:
    -   Training loss showed consistent decrease from \~0.70 to \~0.28
    -   Validation loss exhibited unstable behavior, increasing from
        \~0.70 to \~1.0
    -   Significant divergence between training and validation loss
        after epoch 15
2.  **Accuracy Performance**:
    -   Validation accuracy fluctuated significantly between 0.40 and
        0.70
    -   Best validation accuracy achieved was 0.70 at epoch 5
    -   No clear trend of improving accuracy over time
    -   Final validation accuracy stabilized around 0.60
3.  **Convergence Issues**:
    -   Early stopping triggered at epoch 28 due to lack of improvement
    -   Signs of overfitting emerged after epoch 15 with diverging
        train/val losses
    -   Model showed high variance in validation metrics

### 4.2 Performance Analysis

The experimental results highlight several challenges:

1.  **Model Stability**: The high variance in validation accuracy
    (ranging from 0.40 to 0.70) indicates stability issues in the
    model’s predictions.

2.  **Overfitting Indicators**:

    -   Decreasing training loss while validation loss increases
    -   Growing gap between training and validation metrics
    -   Unstable validation accuracy despite improving training metrics

3.  **Limited Generalization**: The model’s inability to achieve
    consistent validation accuracy above 0.70 suggests limited
    generalization capability.

## 5. Technical Considerations and Limitations

1.  **The pipeline thought process** I first tried the CNN+dropout method, and the final output is 100% correct for each epoch, which is obviously not realistic. So I tried to train with lstm, but the result fluctuates a lot, and the correct rate is low, which is not up to the expectation. Then I used transformer model, the training results can be basically stabilized at about 60%, although the correct rate is not very high, but maintained within my acceptable range. Inspired by the technical line of GPT-2, I used the encoder part of the transformer as a model for training, and the results were still mediocre, so I finally chose the transformer model as the final answer. All the attempted codes were uploaded to my github in the attempt folder.

2.  **Fixed-length Processing**: The system standardizes all inputs to
    130 time steps, which may lose information from longer recordings or
    pad shorter ones.

3.  **Memory Requirements**: The transformer architecture’s quadratic
    attention complexity requires careful batch size management (set to
    8 in implementation).

4.  **GPU Acceleration**: The system is designed to utilize GPU
    acceleration when available but can fall back to CPU processing.

5.  **dataset**:The training set is too small, the training effect is
    poor, there is no standard test set to verify the training effect of
    the model.

## 6. Future Improvements

Several potential improvements could enhance the system:

1.  **Variable Length Processing**: Implement dynamic padding and
    masking to handle variable-length inputs more effectively.

2.  **Additional Features**: Incorporate prosodic features like pitch
    and energy variations.

3.  **Cross-lingual Adaptation**: Develop language-specific models or
    language-agnostic features for better cross-lingual performance.

4.  **dataset**: Avoid overfitting by using different types of official
    datasets that contain more data.

5.  **Hyperparameters**: Hyperparameters need to be further adjusted by
    further training 
    
    
## 7. Conclusion

This implementation demonstrates a practical approach to automated
deception detection in narrated stories. The combination of spectral and
cepstral features processed through a transformer architecture provides
a robust foundation for this challenging task. While the system shows
promise, careful consideration of its limitations and potential biases
is essential for responsible deployment.

## References

1.  Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2.  McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis
    in Python.
3.  PyTorch documentation and tutorials.
4.  Muli,https://zh.d2l.ai/
5.  https://github.com/datawhalechina/llms-from-scratch-cn by DataWhale