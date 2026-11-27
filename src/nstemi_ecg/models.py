from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=-1, keepdim=True)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Weight diagnostically relevant time segments and fuse them with average/max pooling."""

    def __init__(self, channels: int, attention_hidden: int = 64) -> None:
        super().__init__()
        hidden = min(max(16, attention_hidden), channels)
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        average = x.mean(dim=-1)
        maximum = x.amax(dim=-1)
        scores = self.score(x).squeeze(1)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.sum(x * weights.unsqueeze(1), dim=-1)
        gate = self.gate(torch.cat([average, maximum, attended], dim=1))
        fused = gate * attended + (1.0 - gate) * (0.5 * (average + maximum))
        return torch.cat([fused, average, maximum], dim=1), weights


class CrossLeadTransformer(nn.Module):
    """Model inter-lead interactions and aggregate lead tokens through a CLS token."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.10,
        lead_count: int = 12,
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by heads")
        self.project = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.lead_embedding = nn.Parameter(torch.zeros(1, lead_count, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(token_dim)
        self.scale = token_dim ** -0.5
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.lead_embedding, std=0.02)

    def forward(self, lead_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.project(lead_features) + self.lead_embedding
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        cls_output = self.norm(encoded[:, 0])
        lead_tokens = encoded[:, 1:]
        logits = torch.einsum("bd,bld->bl", cls_output, lead_tokens) * self.scale
        lead_weights = torch.softmax(logits, dim=1)
        fused = cls_output + torch.sum(lead_tokens * lead_weights.unsqueeze(-1), dim=1)
        return fused, lead_weights


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        branch_channels = max(8, out_channels // 3)
        self.branch3 = ConvNormAct(out_channels, branch_channels, kernel_size=3)
        self.branch5 = ConvNormAct(out_channels, branch_channels, kernel_size=5)
        self.branch7 = ConvNormAct(out_channels, branch_channels, kernel_size=7)
        fused = branch_channels * 3
        self.fuse = nn.Sequential(
            nn.Conv1d(fused, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            SqueezeExcite1D(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.pre(x)
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        x = self.fuse(x)
        return self.activation(x + residual)


class SharedLeadEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        stage_blocks: list[int] | tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            ConvNormAct(1, base_channels, kernel_size=15, stride=2, padding=7),
            ConvNormAct(base_channels, base_channels, kernel_size=7),
        )
        stages = []
        current = base_channels
        for stage_index, (out_channels, block_count) in enumerate(zip(channels, stage_blocks)):
            blocks = []
            for block_index in range(int(block_count)):
                stride = 2 if block_index == 0 else 1
                blocks.append(MultiScaleResidualBlock(current, out_channels, stride=stride))
                current = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.out_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class LeadAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(16, feature_dim // 4)
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, lead_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(lead_features).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        fused = torch.sum(lead_features * weights.unsqueeze(-1), dim=1)
        return fused, weights


class MSResNetECG(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_length: int,
        clinical_dim: int = 0,
        base_channels: int = 64,
        stage_blocks: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.25,
        use_lead_attention: bool = True,
        clinical_hidden_dim: int = 32,
        temporal_attention: bool = False,
        cross_lead_transformer: bool = False,
        transformer_token_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length
        self.encoder = SharedLeadEncoder(base_channels, stage_blocks)
        encoder_channels = self.encoder.out_channels
        self.temporal_attention = (
            TemporalAttentionPooling(encoder_channels)
            if temporal_attention
            else None
        )
        feature_dim = encoder_channels * (3 if temporal_attention else 2)
        self.lead_norm = nn.LayerNorm(feature_dim)
        self.cross_lead_transformer = (
            CrossLeadTransformer(
                input_dim=feature_dim,
                token_dim=transformer_token_dim,
                layers=transformer_layers,
                heads=transformer_heads,
                dropout=transformer_dropout,
            )
            if cross_lead_transformer
            else None
        )
        self.attention = (
            LeadAttention(feature_dim)
            if use_lead_attention and self.cross_lead_transformer is None
            else None
        )
        fusion_feature_dim = (
            transformer_token_dim
            if self.cross_lead_transformer is not None
            else feature_dim
        )
        self.clinical_dim = clinical_dim
        self.clinical_mlp = None
        fusion_dim = fusion_feature_dim
        if clinical_dim:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(clinical_dim, clinical_hidden_dim),
                nn.LayerNorm(clinical_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            fusion_dim += clinical_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes if num_classes > 2 else 1),
        )

    def forward(self, signal: torch.Tensor, clinical: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch_size, lead_count, length = signal.shape
        if length != self.input_length:
            signal = F.interpolate(
                signal,
                size=self.input_length,
                mode="linear",
                align_corners=False,
            )
        x = signal.reshape(batch_size * lead_count, 1, self.input_length)
        x = self.encoder(x)
        if self.temporal_attention is not None:
            features, _ = self.temporal_attention(x)
        else:
            avg = x.mean(dim=-1)
            maximum = x.amax(dim=-1)
            features = torch.cat([avg, maximum], dim=1)
        features = features.reshape(batch_size, lead_count, -1)
        features = self.lead_norm(features)
        if self.cross_lead_transformer is not None:
            fused, lead_weights = self.cross_lead_transformer(features)
        elif self.attention is None:
            lead_weights = torch.full(
                (batch_size, lead_count),
                1.0 / lead_count,
                device=signal.device,
                dtype=signal.dtype,
            )
            fused = features.mean(dim=1)
        else:
            fused, lead_weights = self.attention(features)
        if self.clinical_mlp is not None:
            if clinical is None or clinical.numel() == 0:
                raise ValueError("This model requires clinical inputs")
            fused = torch.cat([fused, self.clinical_mlp(clinical)], dim=1)
        logits = self.head(fused)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)
        return {"logits": logits, "embedding": fused, "lead_weights": lead_weights}


class SimpleCNNECG(nn.Module):
    """Smaller ECG-only baseline for ablation and rapid debugging."""

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        clinical_dim: int = 0,
        base_channels: int = 32,
        dropout: float = 0.2,
        clinical_hidden_dim: int = 16,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length
        self.features = nn.Sequential(
            ConvNormAct(12, base_channels, kernel_size=15, stride=2),
            nn.MaxPool1d(2),
            ConvNormAct(base_channels, base_channels * 2, kernel_size=7),
            nn.MaxPool1d(2),
            ConvNormAct(base_channels * 2, base_channels * 4, kernel_size=5),
            nn.MaxPool1d(2),
            ConvNormAct(base_channels * 4, base_channels * 8, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        feature_dim = base_channels * 8 + clinical_dim
        self.clinical_dim = clinical_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes if num_classes > 2 else 1),
        )

    def forward(self, signal: torch.Tensor, clinical: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if signal.shape[-1] != self.input_length:
            signal = F.interpolate(signal, size=self.input_length, mode="linear", align_corners=False)
        features = self.features(signal).flatten(1)
        if self.clinical_dim:
            if clinical is None or clinical.numel() == 0:
                raise ValueError("This model requires clinical inputs")
            features = torch.cat([features, clinical], dim=1)
        logits = self.head(features)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)
        lead_weights = torch.full(
            (signal.shape[0], signal.shape[1]),
            1.0 / signal.shape[1],
            device=signal.device,
        )
        return {"logits": logits, "embedding": features, "lead_weights": lead_weights}


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    data_config = config["data"]
    clinical_dim = len(data_config.get("clinical_columns", [])) if data_config.get("use_clinical") else 0
    name = model_config.get("name", "ms_resnet1d")
    common = {
        "num_classes": int(data_config["num_classes"]),
        "input_length": int(data_config["input_length"]),
        "clinical_dim": clinical_dim,
    }
    if name == "ms_resnet1d":
        return MSResNetECG(
            **common,
            base_channels=int(model_config.get("base_channels", 64)),
            stage_blocks=model_config.get("stage_blocks", [2, 2, 2, 2]),
            dropout=float(model_config.get("dropout", 0.25)),
            use_lead_attention=bool(model_config.get("use_lead_attention", True)),
            clinical_hidden_dim=int(model_config.get("clinical_hidden_dim", 32)),
            temporal_attention=bool(model_config.get("temporal_attention", False)),
            cross_lead_transformer=bool(model_config.get("cross_lead_transformer", False)),
            transformer_token_dim=int(model_config.get("transformer_token_dim", 128)),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            transformer_heads=int(model_config.get("transformer_heads", 4)),
            transformer_dropout=float(model_config.get("transformer_dropout", 0.10)),
        )
    if name == "attention_resnet1d":
        return MSResNetECG(
            **common,
            base_channels=int(model_config.get("base_channels", 64)),
            stage_blocks=model_config.get("stage_blocks", [2, 2, 2, 2]),
            dropout=float(model_config.get("dropout", 0.25)),
            use_lead_attention=bool(model_config.get("use_lead_attention", True)),
            clinical_hidden_dim=int(model_config.get("clinical_hidden_dim", 32)),
            temporal_attention=bool(model_config.get("temporal_attention", True)),
            cross_lead_transformer=bool(
                model_config.get("cross_lead_transformer", True)
            ),
            transformer_token_dim=int(model_config.get("transformer_token_dim", 128)),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            transformer_heads=int(model_config.get("transformer_heads", 4)),
            transformer_dropout=float(model_config.get("transformer_dropout", 0.10)),
        )
    if name == "simple_cnn":
        return SimpleCNNECG(
            **common,
            base_channels=int(model_config.get("base_channels", 32)),
            dropout=float(model_config.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown model name: {name}")
