# BDD100K MoE Training Scaffold

Files:
- `bdd100k_moe.py`: metadata preparation + staged PyTorch training
- `requirements.txt`: minimal runtime dependencies

Recommended backbone:
- `convnextv2_tiny.fcmae_ft_in1k`

Reason:
- practical pretrained ConvNeXt-V2 backbone in timm
- strong transfer learning behavior
- simpler and more robust than a heavier ViT/Swin setup for a first thesis experiment

Key outputs from `prepare`:
- `metadata_bundle.json`
- `train.json`
- `val.json`
- `test_fixed.json`

The fixed test JSON is the deterministic test split artifact you requested.
