import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import os
import random
from PIL import Image
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from LRS_DeepLabV3Plus import DeepLabV3Plus_Advanced

# =================== 参数设置 ===================
device_ids = [0, 1, 2, 5]
DEVICE = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
NUM_FOREGROUND = 5
NUM_TOTAL = NUM_FOREGROUND + 1
BATCH_SIZE = 16
EPOCHS = 1000
LR = 1e-4
TARGET_SIZE = (608, 912)

# =================== 数据处理 ===================
def process_image(image_path, target_size=TARGET_SIZE):
    with Image.open(image_path) as img:
        if img.mode == 'RGBA':
            img = _remove_alpha(img)
        elif img.mode in ('L', 'P', 'LA'):
            img = img.convert('RGB')
        orig_w, orig_h = img.size
        tgt_h, tgt_w = target_size
        scale_factor = max(orig_w/tgt_w, orig_h/tgt_h)
        algorithm = Image.LANCZOS if scale_factor > 2 else Image.BILINEAR
        if (orig_w, orig_h) != (tgt_w, tgt_h):
            img = img.resize((tgt_w, tgt_h), algorithm)
        arr = np.asarray(img)
        if img.format == 'BMP' and arr.shape[-1] == 3:
            arr = arr[..., ::-1]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return arr.astype(np.uint8)

def _remove_alpha(img):
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[-1])
    return background

# =================== 数据集 ===================
class MMDataset(Dataset):
    def __init__(self, imgs, masks, use_alb=True):
        self.imgs = imgs
        self.masks = masks
        self.use_alb = use_alb

        self.color_map = {
            "Monoclonal":    (236, 62, 49),
            "Erythroid":     (248, 176, 114),
            "Ggranulocytic": (133, 130, 189),
            "Lymphoid":      (79, 153, 201),
            "Monocytic":     (168, 211, 160),
            "Background":    (0, 0, 0),
        }
        self.color2label = {v: i for i, v in enumerate(self.color_map.values())}

        if self.use_alb:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.Resize(height=608, width=912),  # ⬅️ 强制统一尺寸
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
                ToTensorV2()
            ])
        else:
            self.aug_methods = ['h_flip', 'v_flip', 'rot180', 'original']

    def __len__(self):
        return len(self.imgs)

    def _rgb_to_label(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        mask_label = np.zeros((h, w), dtype=np.int64)
        for rgb, cls_id in self.color2label.items():
            match = np.all(mask_rgb == rgb, axis=-1)
            mask_label[match] = cls_id
        return mask_label

    def generate_boundary_mask(self, mask_label, radius=1):
        h, w = mask_label.shape
        boundary = np.zeros((h, w), dtype=np.uint8)
        kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
        unique_ids = np.unique(mask_label)
        for cls_id in unique_ids:
            cls_mask = (mask_label == cls_id).astype(np.uint8)
            dilation = cv2.dilate(cls_mask, kernel, iterations=1)
            edge = dilation ^ cls_mask
            boundary[edge > 0] = 1
        return boundary

    def _random_augment(self, img, mask, boundary):
        method = random.choice(self.aug_methods)
        if method == 'h_flip':
            return (
                torch.flip(img, dims=[2]),
                torch.flip(mask, dims=[1]),
                torch.flip(boundary, dims=[1])
            )
        elif method == 'v_flip':
            return (
                torch.flip(img, dims=[1]),
                torch.flip(mask, dims=[0]),
                torch.flip(boundary, dims=[0])
            )
        elif method == 'rot180':
            return (
                torch.flip(img, dims=[1,2]),
                torch.flip(mask, dims=[0,1]),
                torch.flip(boundary, dims=[0,1])
            )
        else:
            return img, mask, boundary

    def __getitem__(self, idx):
        img_np = self.imgs[idx]
        mask_np = self.masks[idx]
        mask_label = self._rgb_to_label(mask_np)
        boundary_mask = self.generate_boundary_mask(mask_label, radius=1)

        if self.use_alb:
            transformed = self.transform(image=img_np, mask=mask_label)
            img_t = transformed['image']             # [3,H,W]
            mask_t = transformed['mask'].long()      # [H,W]
            b_t = torch.from_numpy(boundary_mask).float()
        else:
            img_t = torch.from_numpy(img_np).float().permute(2, 0, 1)
            mask_t = torch.from_numpy(mask_label).long()
            b_t = torch.from_numpy(boundary_mask).float()
            img_t, mask_t, b_t = self._random_augment(img_t, mask_t, b_t)

        return img_t, mask_t, b_t

data_dir = "/home/daci3090/data3090/Dataset/Bio/MM/"
img_paths = sorted(os.listdir(os.path.join(data_dir, 'img')), key=lambda x: int(x.split('.')[0]))
mask_paths = sorted(os.listdir(os.path.join(data_dir, 'mask_acc')), key=lambda x: int(x.split('.')[0]))

print("Loading images...")
imgs = [process_image(os.path.join(data_dir, 'img', p)) for p in tqdm(img_paths)]
masks = [process_image(os.path.join(data_dir, 'mask_acc', p)) for p in tqdm(mask_paths)]

dataset = MMDataset(imgs, masks, use_alb=True)
train_len = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =================== 损失函数 ===================
class ComboLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, alpha=0.25, aux_weight=0.4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.aux_weight = aux_weight

    def forward(self, logits, targets, aux_logits=None):
        ce_loss = self.ce(logits, targets)
        logp = -F.log_softmax(logits, dim=1)
        pt = torch.exp(-logp)
        focal_loss = (1 - pt) ** self.gamma * logp
        focal_loss = focal_loss.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).mean()
        smooth = 1e-6
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()
        total_loss = ce_loss + dice_loss + self.alpha * focal_loss
        if aux_logits is not None:
            aux_ce = self.ce(aux_logits, targets)
            total_loss += self.aux_weight * aux_ce
        return total_loss, ce_loss

criterion = ComboLoss(num_classes=NUM_TOTAL).to(DEVICE)

# =================== EMA ===================
class EMA:
    def __init__(self, model, decay=0.99):
        self.shadow = {}
        self.model = model
        self.decay = decay
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

# =================== 模型 ===================
model = DeepLabV3Plus_Advanced(num_classes=NUM_TOTAL).to(DEVICE)
if len(device_ids) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
ema = EMA(model, decay=0.99)

# =================== Metrics ===================
def compute_iou(pred, target, num_total_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls_id in range(num_total_classes):
        pred_inds = (pred == cls_id)
        target_inds = (target == cls_id)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        ious.append(np.nan if union == 0 else intersection / union)
    return ious

def compute_dice(pred, target, num_total_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []
    for cls_id in range(num_total_classes):
        tp = ((pred == cls_id) & (target == cls_id)).sum().item()
        fp = ((pred == cls_id) & (target != cls_id)).sum().item()
        fn = ((pred != cls_id) & (target == cls_id)).sum().item()
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        dice_scores.append(dice)
    return dice_scores

class_names = ["Monoclonal", "Erythroid", "Ggranulocytic", "Lymphoid", "Monocytic"]
def print_metrics_table(phase, iou_list, dice_list, class_names):
    print(f"\n📊 {phase} Metrics")
    print(f"{'Class':<15} {'IoU':>8} {'Dice':>10}")
    print("-" * 35)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {iou_list[i]:>8.4f} {dice_list[i]:>10.4f}")
    print("-" * 35)
    print(f"{'→ Mean':<15} {np.nanmean(iou_list):>8.4f} {np.nanmean(dice_list):>10.4f}\n")
# =================== EarlyStopping ===================
class EarlyStopping:
    def __init__(self, patience=200, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_iou = None
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_iou, val_dice, model):
        if self.best_iou is None or val_iou > self.best_iou:
            self.best_iou = val_iou
            self.best_model = model.state_dict()
            self.counter = 0
            print(f"✅ 验证指标提升，重置计数器")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ Early stop 计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping()

# =================== 训练 ===================
print("========== Start Training ==========")
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_seg_loss = 0, 0
    total_iou = np.zeros(NUM_FOREGROUND)
    total_dice = np.zeros(NUM_FOREGROUND)

    for imgs, seg_masks, _ in tqdm(train_loader, desc=f"[Train {epoch+1}/{EPOCHS}]"):
        imgs, seg_masks = imgs.to(DEVICE), seg_masks.to(DEVICE)
        seg_out, aux_out = model(imgs)
        loss, seg_loss = criterion(seg_out, seg_masks, aux_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()

        preds = seg_out.argmax(dim=1)
        iou_per_class = compute_iou(preds, seg_masks, NUM_TOTAL)
        dice_per_class = compute_dice(preds, seg_masks, NUM_TOTAL)
        total_iou += np.nan_to_num(iou_per_class[:NUM_FOREGROUND])
        total_dice += np.nan_to_num(dice_per_class[:NUM_FOREGROUND])

    avg_iou = total_iou / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    print_metrics_table("Train", avg_iou, avg_dice, class_names)
    print(f"[Train Epoch {epoch+1}] loss={total_loss:.4f} | mIoU={np.nanmean(avg_iou):.4f}, mDice={np.nanmean(avg_dice):.4f}")

    # ========== 验证 ==========
    model.eval()
    ema.apply_shadow()
    val_iou = np.zeros(NUM_FOREGROUND)
    val_dice = np.zeros(NUM_FOREGROUND)
    val_loss = 0

    with torch.no_grad():
        for imgs, seg_masks, _ in tqdm(val_loader, desc=f"[Val {epoch+1}/{EPOCHS}]"):
            imgs, seg_masks = imgs.to(DEVICE), seg_masks.to(DEVICE)
            seg_out, aux_out = model(imgs)
            loss, _ = criterion(seg_out, seg_masks, aux_out)
            val_loss += loss.item()
            preds = seg_out.argmax(dim=1)
            val_iou += np.nan_to_num(compute_iou(preds, seg_masks, NUM_TOTAL)[:NUM_FOREGROUND])
            val_dice += np.nan_to_num(compute_dice(preds, seg_masks, NUM_TOTAL)[:NUM_FOREGROUND])

    avg_val_iou = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    print_metrics_table("Val", avg_val_iou, avg_val_dice, class_names)
    print(f"[Val Epoch {epoch+1}] val_loss={val_loss:.4f} | mIoU={np.nanmean(avg_val_iou):.4f}, mDice={np.nanmean(avg_val_dice):.4f}")

    early_stopper(np.nanmean(avg_val_iou), np.nanmean(avg_val_dice), model)
    if early_stopper.early_stop:
        print("🛑 Early stopping.")
        break

# 保存最优模型
torch.save(early_stopper.best_model, "models/best_model.pth")
print("✅ 模型已保存至 models/best_model.pth")
