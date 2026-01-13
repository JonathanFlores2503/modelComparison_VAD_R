import sys
import os
import torch
import hydra
from omegaconf import OmegaConf
from torchmetrics import AUROC


REPO_ROOT = "/media/pc/MainWork/Codes/TransLowNet_enero_2026/AnomalyClip"
SRC_ROOT = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

DATA_YAML = f"{REPO_ROOT}/configs/data/ucfcrime.yaml"
MODEL_YAML = f"{REPO_ROOT}/configs/model/anomaly_clip_ucfcrime.yaml"

CKPT_PATH = f"{REPO_ROOT}/checkpoints/epoch_49_step_1250.ckpt"
NCENTROID_PATH = f"{REPO_ROOT}/checkpoints/ncentroid.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg_data = OmegaConf.load(DATA_YAML)
cfg_model = OmegaConf.load(MODEL_YAML)


cfg_model.data = cfg_data
OmegaConf.resolve(cfg_model)


datamodule = hydra.utils.instantiate(cfg_data)
model = hydra.utils.instantiate(cfg_model).to(DEVICE)


ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"], strict=True)


net = model.net
net.eval()

# ------------------------------------------------------------
# NCENTROID
# ------------------------------------------------------------
ncentroid = torch.load(NCENTROID_PATH, map_location=DEVICE).to(DEVICE)

# ------------------------------------------------------------
# DATALOADER
# ------------------------------------------------------------
datamodule.setup(stage="test")
test_loader = datamodule.test_dataloader()
normal_id = datamodule.hparams.normal_id

# ------------------------------------------------------------
# EVALUACIÓN FRAME-LEVEL
# ------------------------------------------------------------
auroc = AUROC(task="binary")
scores, labels_all = [], []

with torch.no_grad():
    for batch in test_loader:
        image_features, labels, _, segment_size, _ = batch
        image_features = image_features.to(DEVICE)
        labels = labels.squeeze(0).to(DEVICE)

        _, abnormal_scores = net(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode=True,
        )

        scores.append(abnormal_scores.cpu())
        labels_all.append(labels.cpu())

scores = torch.cat(scores)
labels_all = torch.cat(labels_all)
labels_bin = torch.where(labels_all == normal_id, 0, 1)

auc = auroc(scores, labels_bin)
print(f"\n✅ AUC ROC (frame-level): {auc.item():.4f}\n")
