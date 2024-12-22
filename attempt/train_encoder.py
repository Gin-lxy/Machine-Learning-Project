import numpy as np
import pandas as pd
import librosa 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_audio_features(file_path, fixed_length=130):
    """
    Enhanced audio feature extraction with additional features
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # 1. Enhanced Mel spectrogram with more bands
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0,
            fmin=20,
            fmax=8000  # Expanded frequency range
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. Enhanced MFCC features
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=40,
            n_fft=2048,
            hop_length=512
        )
        
        # 3. Add delta and acceleration features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # 4. Add spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # 5. Add rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zero_crossings = librosa.zero_crossings(y, pad=False)
        zero_crossing_rate = np.mean(zero_crossings)
        
        # Handle length uniformity with improved padding
        def pad_or_truncate(array, target_length):
            if len(array.shape) == 1:
                array = array.reshape(1, -1)
            current_length = array.shape[1]
            if current_length > target_length:
                return array[:, :target_length]
            else:
                pad_width = ((0, 0), (0, target_length - current_length))
                return np.pad(array, pad_width, mode='reflect')
        
        # Process all features to uniform length
        mel_spec_db = pad_or_truncate(mel_spec_db, fixed_length)
        mfccs = pad_or_truncate(mfccs, fixed_length)
        mfcc_delta = pad_or_truncate(mfcc_delta, fixed_length)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, fixed_length)
        spectral_centroids = pad_or_truncate(spectral_centroids.reshape(1, -1), fixed_length)
        spectral_rolloff = pad_or_truncate(spectral_rolloff.reshape(1, -1), fixed_length)
        spectral_bandwidth = pad_or_truncate(spectral_bandwidth.reshape(1, -1), fixed_length)
        
        # Improved feature standardization with robust scaling
        scaler = RobustScaler()
        mel_spec_db = scaler.fit_transform(mel_spec_db)
        mfccs = scaler.fit_transform(mfccs)
        mfcc_delta = scaler.fit_transform(mfcc_delta)
        mfcc_delta2 = scaler.fit_transform(mfcc_delta2)
        spectral_features = scaler.fit_transform(np.vstack([
            spectral_centroids,
            spectral_rolloff,
            spectral_bandwidth
        ]))
        
        return {
            'mel_spec': mel_spec_db.astype(np.float32),
            'mfcc': np.vstack([mfccs, mfcc_delta, mfcc_delta2]).astype(np.float32),
            'spectral': spectral_features.astype(np.float32),
            'rhythm': np.array([tempo, zero_crossing_rate], dtype=np.float32)
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        
        if in_features != out_features:
            self.downsample = nn.Linear(in_features, out_features)
        else:
            self.downsample = None
            
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.gelu(out)
        
        return out

class ImprovedTransformerModel(nn.Module):
    def __init__(self, mel_channels=128, mfcc_channels=120, spectral_channels=3,
                 d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Feature embedding layers with layer normalization
        self.mel_embed = nn.Sequential(
            nn.Linear(mel_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.mfcc_embed = nn.Sequential(
            nn.Linear(mfcc_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.spectral_embed = nn.Sequential(
            nn.Linear(spectral_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Rhythm features embedding
        self.rhythm_embed = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = nn.Sequential(
            PositionalEncoding(d_model),
            nn.Dropout(dropout)
        )
        
        # Attention weights for global pooling
        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        
        # Bidirectional Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Improved classification head with residual connections
        self.classifier = nn.Sequential(
            ResidualBlock(d_model, dim_feedforward),
            ResidualBlock(dim_feedforward, dim_feedforward // 2),
            ResidualBlock(dim_feedforward // 2, dim_feedforward // 4),
            nn.Linear(dim_feedforward // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, mel_spec, mfcc, spectral, rhythm):
        # Process features
        mel_embedded = self.mel_embed(mel_spec.transpose(1, 2))
        mfcc_embedded = self.mfcc_embed(mfcc.transpose(1, 2))
        spectral_embedded = self.spectral_embed(spectral.transpose(1, 2))
        rhythm_embedded = self.rhythm_embed(rhythm).unsqueeze(1)
        
        # Combine features
        combined = torch.cat([
            mel_embedded,
            mfcc_embedded,
            spectral_embedded,
            rhythm_embedded.expand(-1, mel_embedded.size(1), -1)
        ], dim=-1)
        
        # Apply positional encoding
        encoded = self.pos_encoder(combined)
        
        # Transformer processing
        transformer_output = self.transformer_encoder(encoded)
        
        # Global attention pooling
        attention_weights = F.softmax(
            self.attention_weights(transformer_output),
            dim=1
        )
        pooled = torch.sum(transformer_output * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class DeceptionDataset(torch.utils.data.Dataset):
    def __init__(self, features_list, labels):
        self.features = features_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        mel_spec = torch.FloatTensor(features['mel_spec']).unsqueeze(0)
        mfcc = torch.FloatTensor(features['mfcc'])
        spectral = torch.FloatTensor(features['spectral'])
        rhythm = torch.FloatTensor(features['rhythm'])
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'spectral': spectral,
            'rhythm': rhythm,
            'label': label
        }

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        
        # Dynamically adjust alpha based on class imbalance
        batch_pos = torch.sum(targets)
        batch_size = targets.size(0)
        pos_weight = batch_pos / batch_size
        dynamic_alpha = torch.where(
            targets == 1,
            1 - pos_weight,
            pos_weight
        )
        
        focal_loss = dynamic_alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, epochs=150, patience=15):
    model = model.to(device)
    criterion = AdaptiveFocalLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_metrics': [],
        'learning_rates': []
    }
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for batch in pbar:
                mel_spec = batch['mel_spec'].to(device)
                mfcc = batch['mfcc'].to(device)
                spectral = batch['spectral'].to(device)
                rhythm = batch['rhythm'].to(device)
                labels = batch['label'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(mel_spec, mfcc, spectral, rhythm)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rates'].append(current_lr)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spec'].to(device)
                mfcc = batch['mfcc'].to(device)
                spectral = batch['spectral'].to(device)
                rhythm = batch['rhythm'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(mel_spec, mfcc, spectral, rhythm)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        accuracy = accuracy_score(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(metrics)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': metrics
            }, 'best_model.pth')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    return history

def plot_training_history(history):
    """
    Enhanced visualization of the training process
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History', fontsize=16, y=1.02)
    
    # Plot loss
    axes[0,0].plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    axes[0,0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[0,0].set_title('Loss Evolution')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['green', 'orange', 'purple', 'brown']
    
    for metric, color in zip(metrics, colors):
        values = [m[metric] for m in history['val_metrics']]
        axes[0,1].plot(values, label=metric.capitalize(), color=color, alpha=0.7)
    
    axes[0,1].set_title('Validation Metrics')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Score')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot learning rate
    axes[1,0].plot(history['learning_rates'], label='Learning Rate', 
                   color='purple', alpha=0.7)
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Add convergence analysis
    window = 5
    train_smooth = pd.Series(history['train_loss']).rolling(window=window).mean()
    val_smooth = pd.Series(history['val_loss']).rolling(window=window).mean()
    
    axes[1,1].plot(train_smooth, label='Smoothed Train Loss', 
                   color='blue', alpha=0.7)
    axes[1,1].plot(val_smooth, label='Smoothed Val Loss', 
                   color='red', alpha=0.7)
    axes[1,1].set_title(f'Loss Convergence (Window={window})')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Smoothed Loss')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader):
    """
    Comprehensive model evaluation
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            mel_spec = batch['mel_spec'].to(device)
            mfcc = batch['mfcc'].to(device)
            spectral = batch['spectral'].to(device)
            rhythm = batch['rhythm'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(mel_spec, mfcc, spectral, rhythm)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=['False', 'True'],
        yticklabels=['False', 'True']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'probabilities': np.array(all_probs),
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels)
    }

def main():
    # Configuration parameters
    config = {
        'csv_path': 'CBU0521DD_stories_attributes.csv',  # 确保文件名正确
        'audio_dir': 'CBU0521DD_stories',  # 简化路径
        'batch_size': 16,
        'fixed_length': 130,
        'epochs': 30,
        'patience': 10,
        'seed': 42,
        'model_params': {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1
        }
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
    
    # Load and process data
    print("Loading and processing audio files...")
    
    try:
        df = pd.read_csv(config['csv_path'])
        print(f"Found {len(df)} entries in CSV file")
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {config['csv_path']}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    
    features_list = []
    labels = []
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_path = os.path.join(config['audio_dir'], row['filename'])
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
                
            features = load_audio_features(audio_path, fixed_length=config['fixed_length'])
            
            if features is not None:
                features_list.append(features)
                labels.append(1 if 'True Story' in row['Story_type'] else 0)
            else:
                errors.append(f"Failed to process: {audio_path}")
                
        except Exception as e:
            errors.append(f"Error processing {row['filename']}: {str(e)}")
            continue
    
    # Check if we have any data
    if len(features_list) == 0:
        print("Error: No audio files were successfully processed!")
        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(error)
        return
    
    print(f"\nSuccessfully processed {len(features_list)} out of {len(df)} audio files")
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(error)
        if len(errors) > 5:
            print(f"...and {len(errors)-5} more errors")
    
    # Split data
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_list, labels, 
            test_size=0.3, 
            stratify=labels,
            random_state=config['seed']
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=config['seed']
        )
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return
    
    print("\nData split sizes:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Create data loaders
    train_dataset = DeceptionDataset(X_train, y_train)
    val_dataset = DeceptionDataset(X_val, y_val)
    test_dataset = DeceptionDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True
    )
    
    # Rest of the code remains the same...
    
    # Initialize model
    model = ImprovedTransformerModel(**config['model_params'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    history = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    best_model = ImprovedTransformerModel(**config['model_params'])
    checkpoint = torch.load('best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(best_model, test_loader)
    
    print("\nTraining complete! Check 'training_history.png' and 'confusion_matrix.png' for visualizations.")

if __name__ == "__main__":
    main()