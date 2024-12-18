import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#  Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio_features(file_path, fixed_length=130):
    """
    Load audio and extract enhanced features
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=40,
            n_fft=2048,
            hop_length=512
        )
        
        # Add delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Handle length uniformity
        def pad_or_truncate(array, target_length):
            current_length = array.shape[1]
            if current_length > target_length:
                return array[:, :target_length]
            else:
                pad_width = ((0, 0), (0, target_length - current_length))
                return np.pad(array, pad_width, mode='reflect')
        
        # Process feature lengths
        mel_spec_db = pad_or_truncate(mel_spec_db, fixed_length)
        mfccs = pad_or_truncate(mfccs, fixed_length)
        mfcc_delta = pad_or_truncate(mfcc_delta, fixed_length)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, fixed_length)
        
        # Feature standardization
        scaler = StandardScaler()
        mel_spec_db = scaler.fit_transform(mel_spec_db)
        mfccs = scaler.fit_transform(mfccs)
        mfcc_delta = scaler.fit_transform(mfcc_delta)
        mfcc_delta2 = scaler.fit_transform(mfcc_delta2)
        
        return {
            'mel_spec': mel_spec_db.astype(np.float32),
            'mfcc': np.vstack([mfccs, mfcc_delta, mfcc_delta2]).astype(np.float32)
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerDecoderOnlyModel(nn.Module):
    def __init__(self, mel_channels=128, mfcc_channels=120,d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Feature embedding layers
        self.mel_embed = nn.Sequential(
            nn.Linear(mel_channels, d_model),
            nn.LayerNorm(d_model)
        )
        self.mfcc_embed = nn.Sequential(
            nn.Linear(mfcc_channels, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # 输出logits
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, mel_spec, mfcc):
        # Process features with embeddings
        mel_embedded = self.mel_embed(mel_spec.squeeze(1).transpose(1, 2))
        mfcc_embedded = self.mfcc_embed(mfcc.transpose(1, 2))
        
        # Apply positional encoding
        mel_encoded = self.pos_encoder(mel_embedded)
        mfcc_encoded = self.pos_encoder(mfcc_embedded)
        
        # Combine features
        tgt = mel_encoded
        memory = mfcc_encoded
        
        # Generate attention mask for causal attention
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Apply Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Global average pooling
        pooled = torch.mean(decoder_output, dim=1)
        
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
        mel_spec = torch.FloatTensor(self.features[idx]['mel_spec']).unsqueeze(0)
        mfcc = torch.FloatTensor(self.features[idx]['mfcc'])
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'label': label
        }

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, epochs=150, patience=15):
    model = model.to(device)
    criterion = FocalLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_accuracy': [],
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
                labels = batch['label'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(mel_spec, mfcc)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rates'].append(current_lr)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spec'].to(device)
                mfcc = batch['mfcc'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(mel_spec, mfcc)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], 'f'Loss: {avg_train_loss:.4f}, 'f'Val Loss: {avg_val_loss:.4f}, 'f'Val Accuracy: {val_accuracy:.4f}')
    
    return history
def plot_training_history(history):
    """
    Visualization of the training process
    """
    # Creating Subgraphs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    axes[0,0].plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    axes[0,0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # Plot accuracy
    axes[0,1].plot(history['val_accuracy'], label='Validation Accuracy', color='green', alpha=0.7)
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Plot learning rate
    axes[1,0].plot(history['learning_rates'], label='Learning Rate', color='purple', alpha=0.7)
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Keep the last subplot empty for future metrics
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix):
    """
    Drawing the confusion matrix
    """
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

def main():
    # Configuration parameters
    config = {
        'csv_path': 'CBU0521DD_stories_attributes.csv',
        'audio_dir': 'Deception/Deception-main/CBU0521DD_stories',
        'batch_size': 16,
        'fixed_length': 130,
        'epochs': 30,
        'patience': 15,
        'seed': 42,
        'model_params': {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1
        }
    }
    
    # Setting the random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load data
    print("Loading and processing audio files...")
    df = pd.read_csv(config['csv_path'])
    features_list = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = f"{config['audio_dir']}/{row['filename']}"
        features = load_audio_features(audio_path, fixed_length=config['fixed_length'])
        
        if features is not None:
            features_list.append(features)
            labels.append(1 if 'True Story' in row['Story_type'] else 0)
    
    print(f"\nProcessed {len(features_list)} audio files successfully")
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # Dataset segmentation
    X_train, X_val, y_train, y_val = train_test_split(
        features_list, labels, 
        test_size=0.2, 
        random_state=config['seed'],
        stratify=labels
    )
    
    # Creating a Data Loader
    train_dataset = DeceptionDataset(X_train, y_train)
    val_dataset = DeceptionDataset(X_val, y_val)
    
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
    
    # initialise model
    model = TransformerDecoderOnlyModel(**config['model_params'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # train model
    print("\nStarting training...")
    history = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Visualize the training process
    plot_training_history(history)
    
    # Load the best model for final evaluation
    print("\nLoading best model for final evaluation...")
    best_model = TransformerDecoderOnlyModel(**config['model_params'])
    checkpoint = torch.load('best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    best_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            mel_spec = batch['mel_spec'].to(device)
            mfcc = batch['mfcc'].to(device)
            labels = batch['label'].to(device)
            
            outputs = best_model(mel_spec, mfcc)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and display final metrics
    final_accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Drawing the confusion matrix
    plot_confusion_matrix(conf_matrix)
    
    print("\nTraining complete! Results saved as 'training_history.png' and 'confusion_matrix.png'")

if __name__ == "__main__":
    main()