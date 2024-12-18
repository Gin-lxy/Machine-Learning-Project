import torch
import librosa
import numpy as np
from torch import nn

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

class AudioTransformer(nn.Module):
    def __init__(self, mel_channels=128, mfcc_channels=20, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.mel_embed = nn.Linear(mel_channels, d_model)
        self.mfcc_embed = nn.Linear(mfcc_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, mel_spec, mfcc):
        mel_embedded = self.mel_embed(mel_spec.squeeze(1).transpose(1, 2))
        mel_encoded = self.pos_encoder(mel_embedded)
        
        mfcc_embedded = self.mfcc_embed(mfcc.transpose(1, 2))
        mfcc_encoded = self.pos_encoder(mfcc_embedded)
        
        combined = torch.cat([mel_encoded, mfcc_encoded], dim=1)
        transformer_output = self.transformer_encoder(combined)
        pooled = torch.mean(transformer_output, dim=1)
        output = self.classifier(pooled)
        
        return output

class StoryPredictor:
    def __init__(self, model_path='best_model.pth'):
        """
        初始化预测器
        model_path: 训练好的模型权重路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioTransformer()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def process_audio(self, audio_path, fixed_length=130):
        """
        处理音频文件并提取特征
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path)
            
            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # 统一长度处理
            def pad_or_truncate(array, target_length):
                current_length = array.shape[1]
                if current_length > target_length:
                    return array[:, :target_length]
                else:
                    pad_width = ((0, 0), (0, target_length - current_length))
                    return np.pad(array, pad_width, mode='constant')
            
            mel_spec_db = pad_or_truncate(mel_spec_db, fixed_length)
            mfccs = pad_or_truncate(mfccs, fixed_length)
            
            return {
                'mel_spec': mel_spec_db.astype(np.float32),
                'mfcc': mfccs.astype(np.float32)
            }
            
        except Exception as e:
            print(f"处理音频文件时出错: {str(e)}")
            return None
    
    def predict(self, audio_path):
        """
        预测故事的真实性
        返回: (预测结果, 置信度)
        """
        # 处理音频
        features = self.process_audio(audio_path)
        if features is None:
            return None, None
        
        # 准备数据
        mel_spec = torch.FloatTensor(features['mel_spec']).unsqueeze(0).unsqueeze(0)
        mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0)
        
        # 移动数据到设备
        mel_spec = mel_spec.to(self.device)
        mfcc = mfcc.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(mel_spec, mfcc)
            probability = output.item()
            prediction = probability > 0.5
        
        return prediction, probability

def predict_batch(predictor, audio_files):
    """
    批量预测多个音频文件
    """
    results = []
    for audio_file in audio_files:
        prediction, confidence = predictor.predict(audio_file)
        results.append({
            'file': audio_file,
            'prediction': '真实' if prediction else '虚假',
            'confidence': confidence
        })
    return results

# 使用示例
def main():
    # 初始化预测器
    predictor = StoryPredictor('best_model.pth')
    
    # 单个文件预测
    audio_path = "test_audio.wav"  # 替换为你的音频文件路径
    prediction, confidence = predictor.predict(audio_path)
    
    if prediction is not None:
        result = "真实" if prediction else "虚假"
        print(f"音频文件: {audio_path}")
        print(f"预测结果: 这个故事很可能是{result}的")
        print(f"置信度: {confidence:.2%}")
    
    # 批量预测
    audio_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav"
    ]
    
    print("\n批量预测结果:")
    results = predict_batch(predictor, audio_files)
    for result in results:
        print(f"\n文件: {result['file']}")
        print(f"预测结果: {result['prediction']}")
        print(f"置信度: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()