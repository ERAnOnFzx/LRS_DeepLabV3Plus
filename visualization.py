import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from LRS_DeepLabV3Plus import DeepLabV3Plus_Advanced
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
from collections import OrderedDict

NUM_CLASSES = 6
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("load data.")
data_dir = "/home/daci3090/data3090/Dataset/Bio/MM/"
img_dir = data_dir + '/img/'
labels_path = data_dir + '/mask_acc/'
test_ = img_dir + '350.jpg' # Change visualization image here.
test_m = labels_path + '350.png'

# Load model
device_ids = [3, 7]
model = DeepLabV3Plus_Advanced(num_classes=6)
state_dict = torch.load("models/best_model.pth")

# 如果 state_dict 的 key 是带 "module." 的，去掉它
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # 去掉 module.
    new_state_dict[name] = v

# 加载处理过的 state_dict
model.load_state_dict(new_state_dict)

# 设置为评估模式
model.eval()

def process_image(image_path, target_size=(608, 912)):
    """
    统一图像处理流程：
    1. 读取任意格式图像
    2. 转换为RGB模式
    3. 智能调整尺寸
    4. 保证通道顺序为RGB
    参数：
    target_size - 目标尺寸 (height, width)
    返回：
    (1216, 1824, 3) 的uint8数组
    """
    # 统一加载和预处理
    with Image.open(image_path) as img:
        # 处理特殊模式
        if img.mode == 'RGBA':
            img = _remove_alpha(img)
        elif img.mode in ('L', 'P', 'LA'):
            img = img.convert('RGB')
        
        # 获取原始尺寸
        orig_width, orig_height = img.size
        target_height, target_width = target_size

        # 动态选择缩放算法
        scale_factor = max(orig_width/target_width, orig_height/target_height)
        algorithm = Image.LANCZOS if scale_factor > 2 else Image.BILINEAR

        # 执行尺寸调整
        if (orig_width, orig_height) != (target_width, target_height):
            img = img.resize((target_width, target_height), algorithm)
        
        # 转换为numpy数组
        arr = np.asarray(img)
        
        # 特殊格式处理 (BMP的BGR转换)
        if img.format == 'BMP' and arr.shape[-1] == 3:
            arr = arr[..., ::-1]  # BGR转RGB

    # 最终维度验证
    if arr.ndim == 2:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    
    return arr.astype(np.uint8)

def _remove_alpha(img):
    """处理透明通道：添加白色背景"""
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[-1])  # 用alpha通道作为mask
    return background

class MMDataset(Dataset):
    def __init__(self, imgs, masks, use_alb=True, enhanced=False):
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
            if enhanced:
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
                self.transform = A.Compose([
                A.HorizontalFlip(p=0),
                A.VerticalFlip(p=0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0),
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
color_map = {
        "Monoclonal": (236, 62, 49),
        "Erythroid": (248, 176, 114),
        "Ggranulocytic": (133, 130, 189),
        "Lymphoid": (79, 153, 201),
        "Monocytic": (168, 211, 160)
    }

color_map = [(236, 62, 49),(248, 176, 114),(133, 130, 189),(79, 153, 201),(168, 211, 160),(0, 0, 0)]
def decode_segmap(pred_mask, colormap):
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        rgb_mask[pred_mask == class_idx] = color
    return rgb_mask

plt.axis('off')
plt.imshow(np.array([process_image(test_)])[0])
plt.savefig("visualization_img.png",bbox_inches='tight', pad_inches=0, dpi=300)
plt.axis('off')
plt.imshow(np.array([process_image(test_m)])[0])
plt.savefig("visualization_mask.png",bbox_inches='tight', pad_inches=0, dpi=300)
dataset = MMDataset(np.array([process_image(test_)]), np.array([process_image(test_m)]), enhanced=False)

loader = DataLoader(dataset, batch_size=1)
model = model.to(f'cuda:{device_ids[0]}')
if torch.cuda.device_count() > 1:
    print(f"使用GPU {device_ids} 进行推理")
    model = nn.DataParallel(model, device_ids=device_ids)
output = None
for img, mask, b_t in loader:
    img, mask = img.to(DEVICE), mask.to(DEVICE)
    mask = mask.long()  # 将mask转换为类别标签
    outputs = model(img)[0]
preds = outputs.argmax(1)

print(preds.shape)

plt.imshow(decode_segmap(preds[0].cpu().numpy(), color_map))
plt.axis('off')
plt.savefig("visualization_pred.png", bbox_inches='tight', pad_inches=0, dpi=300)

def count_elements_np(arr):
    arr = arr.flatten()
    unique_elements, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_elements, counts))

# 统计各类别像素个数
print(count_elements_np(preds[0].cpu().numpy()))

print(count_elements_np(mask.cpu().numpy()))