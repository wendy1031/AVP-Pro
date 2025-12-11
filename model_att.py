import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): 
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_weights = self.softmax(attention_scores / (k.size(-1) ** 0.5))
        weighted_features = torch.bmm(attention_weights, v)
        return torch.mean(weighted_features, dim=1)

class ParallelFeatureExtractorWithAttention(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, dropout_rate=0.3):
        super(ParallelFeatureExtractorWithAttention, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.cnn_attention = SelfAttention(cnn_out_channels)
        self.cnn_branch_output_dim = cnn_out_channels

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.bilstm_attention = SelfAttention(lstm_hidden_dim * 2)
        self.bilstm_branch_output_dim = lstm_hidden_dim * 2

    def forward(self, sequence_embedding):
        # CNN Branch
        cnn_in = sequence_embedding.permute(0, 2, 1) 
        cnn_out = self.cnn(cnn_in)
        cnn_out = F.relu(cnn_out)
        cnn_out_permuted = cnn_out.permute(0, 2, 1) 
        v_cnn = self.cnn_attention(cnn_out_permuted)

        # BiLSTM Branch
        lstm_out, _ = self.bilstm(sequence_embedding) 
        v_bilstm = self.bilstm_attention(lstm_out)
        return v_cnn, v_bilstm

class AVP_HNCL_v3(nn.Module):
    def __init__(self, esm_dim, additional_dim, cnn_out_channels, lstm_hidden_dim, num_classes, dropout_rate=0.42):
        super(AVP_HNCL_v3, self).__init__()
        
        fused_input_dim = esm_dim + additional_dim
        self.parallel_extractor = ParallelFeatureExtractorWithAttention(fused_input_dim, cnn_out_channels, lstm_hidden_dim, dropout_rate)
        
        cnn_feature_dim = self.parallel_extractor.cnn_branch_output_dim
        bilstm_feature_dim = self.parallel_extractor.bilstm_branch_output_dim

        # Adaptive Gating
        self.gating_network = nn.Sequential(
            nn.Linear(cnn_feature_dim + bilstm_feature_dim, 1),
            nn.Sigmoid()
        )
        self.cnn_dim_matcher = nn.Linear(cnn_feature_dim, bilstm_feature_dim)
        classifier_input_dim = bilstm_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.BatchNorm1d(classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        self.embedding_dim = classifier_input_dim

    def forward(self, esm_sequence_embedding, additional_features):
        seq_len = esm_sequence_embedding.size(1)
        expanded_additional_features = additional_features.unsqueeze(1).expand(-1, seq_len, -1)
        fused_sequence_embedding = torch.cat([esm_sequence_embedding, expanded_additional_features], dim=2)
        
        v_cnn, v_bilstm = self.parallel_extractor(fused_sequence_embedding)
        v_cnn_matched = self.cnn_dim_matcher(v_cnn)
        
        lambda_gate = self.gating_network(torch.cat([v_cnn, v_bilstm], dim=1))
        final_embedding = lambda_gate * v_cnn_matched + (1 - lambda_gate) * v_bilstm
        
        logits = self.classifier(final_embedding)
        return logits, final_embedding