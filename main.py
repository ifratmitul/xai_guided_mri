import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import glob
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    DATA_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    OUTPUT_DIR = 'guided_binary_classification'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
    
    # Binary Classification
    CLASSES = ['No Tumor', 'Tumor Present']
    NUM_CLASSES = 2
    TUMOR_THRESHOLD = 100
    
    # Model
    INPUT_CHANNELS = 3  # T1CE, FLAIR, T2
    IMG_SIZE = 224
    PRETRAINED = True
    USE_MODALITIES = ['t1ce', 'flair', 't2']
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Dual Loss
    ALPHA = 0.1  # Weight for explanation loss
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Visualization
    SAVE_HEATMAPS_EVERY_N_EPOCHS = 2
    HEATMAPS_PER_EPOCH = 3
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

# Create directories
for dir_path in [Config.MODEL_DIR, Config.LOG_DIR, Config.VIZ_DIR, Config.HEATMAP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# ==================== DATASET ====================
class BinaryBraTSDataset(Dataset):
    """Binary Classification Dataset with Segmentation Masks"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        print(f"Loading BraTS dataset from {data_dir}")
        self._load_dataset()
        print(f"Total samples: {len(self.samples)}")
        self._print_distribution()
    
    def _load_dataset(self):
        """Load all slices from all cases"""
        case_dirs = sorted(glob.glob(os.path.join(self.data_dir, "BraTS20_*")))
        
        for case_dir in case_dirs:
            case_name = os.path.basename(case_dir)
            seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
            
            if not os.path.exists(seg_path):
                seg_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
                if not os.path.exists(seg_path):
                    continue
            
            try:
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()
                
                for slice_idx in range(seg_data.shape[2]):
                    slice_seg = seg_data[:, :, slice_idx]
                    
                    # Count tumor voxels
                    tumor_voxels = np.sum((slice_seg == 1) | (slice_seg == 2) | (slice_seg == 4))
                    
                    # Binary label
                    if tumor_voxels >= Config.TUMOR_THRESHOLD:
                        label = 1  # Tumor present
                    else:
                        label = 0  # No tumor
                    
                    self.samples.append({
                        'case_dir': case_dir,
                        'case_name': case_name,
                        'slice_idx': slice_idx,
                        'label': label,
                        'tumor_voxels': tumor_voxels
                    })
            
            except Exception as e:
                print(f"Error loading {case_name}: {e}")
                continue
    
    def _print_distribution(self):
        """Print class distribution"""
        labels = [s['label'] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        
        print("\nClass Distribution:")
        for label, count in zip(unique, counts):
            class_name = Config.CLASSES[label]
            percentage = (count / len(labels)) * 100
            print(f"  {class_name} (label={label}): {count} samples ({percentage:.1f}%)")
    
    def load_mri_slice(self, case_dir, case_name, slice_idx):
        """Load 3-channel MRI slice"""
        channels = []
        
        for modality in Config.USE_MODALITIES:
            file_path = os.path.join(case_dir, f"{case_name}_{modality}.nii")
            if not os.path.exists(file_path):
                file_path = os.path.join(case_dir, f"{case_name}_{modality}.nii.gz")
            
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            channel = data[:, :, slice_idx]
            
            # Normalize
            if channel.max() > channel.min():
                channel = (channel - channel.min()) / (channel.max() - channel.min())
            
            channels.append(channel)
        
        image_3ch = np.stack(channels, axis=-1).astype(np.float32)
        return image_3ch
    
    def load_segmentation_mask(self, case_dir, case_name, slice_idx):
        """Load binary segmentation mask"""
        seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
        
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata()
        slice_seg = seg_data[:, :, slice_idx]
        
        # Binary mask: any tumor → 1
        binary_mask = (slice_seg > 0).astype(np.float32)
        return binary_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self.load_mri_slice(
            sample['case_dir'],
            sample['case_name'],
            sample['slice_idx']
        )
        
        # Load mask
        mask = self.load_segmentation_mask(
            sample['case_dir'],
            sample['case_name'],
            sample['slice_idx']
        )
        
        # Convert to PIL and resize
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_pil = image_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE), Image.BILINEAR)
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Resize mask
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_pil) / 255.0).float()
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image_tensor, label, mask_tensor

# ==================== MODEL WITH GRADCAM ====================
class BinaryClassifierWithGradCAM(nn.Module):
    """Binary Classifier with Grad-CAM for attention alignment"""
    
    def __init__(self, pretrained=True):
        super(BinaryClassifierWithGradCAM, self).__init__()
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Feature extractor (all layers except FC)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        # For Grad-CAM
        self.gradients = None
        self.target_layer_index = -2  # layer4 (last conv block)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def compute_gradcam(self, image):
        """Compute Grad-CAM heatmap"""
        self.eval()
        image = image.clone().detach().requires_grad_(True)
        
        # Forward through features to get activations
        x = image
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.target_layer_index:
                activations = x
                h = x.register_hook(self.activations_hook)
        
        # Complete forward pass
        output = self.classifier(x)
        
        # Backward
        self.zero_grad()
        output.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample
        cam = F.interpolate(cam, size=(Config.IMG_SIZE, Config.IMG_SIZE), 
                          mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach()

# ==================== LOSS FUNCTIONS ====================
def dice_loss(pred, target, epsilon=1e-6):
    """Dice loss for attention alignment"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)
    
    return 1 - (2 * intersection + epsilon) / (union + epsilon)

# ==================== DATA SPLITTING ====================
def create_patient_level_splits(dataset):
    """Split by patient"""
    case_samples = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        case_samples[sample['case_name']].append(idx)
    
    all_cases = list(case_samples.keys())
    random.shuffle(all_cases)
    
    n_cases = len(all_cases)
    n_train = int(n_cases * Config.TRAIN_SPLIT)
    n_val = int(n_cases * Config.VAL_SPLIT)
    
    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:n_train + n_val]
    test_cases = all_cases[n_train + n_val:]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for case in train_cases:
        train_indices.extend(case_samples[case])
    for case in val_cases:
        val_indices.extend(case_samples[case])
    for case in test_cases:
        test_indices.extend(case_samples[case])
    
    print(f"\nPatient-Level Split:")
    print(f"  Train: {len(train_cases)} patients, {len(train_indices)} slices")
    print(f"  Val:   {len(val_cases)} patients, {len(val_indices)} slices")
    print(f"  Test:  {len(test_cases)} patients, {len(test_indices)} slices")
    
    return train_indices, val_indices, test_indices

# ==================== VISUALIZATION ====================
def save_heatmap_comparison(image, mask, gradcam, pred_class, true_class, save_path):
    """Save 3-panel visualization"""
    plt.figure(figsize=(12, 4))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    t1ce = img[:, :, 0]
    
    # Panel 1: Input
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image", fontsize=14)
    plt.axis('off')
    
    # Panel 2: Expert
    plt.subplot(1, 3, 2)
    plt.imshow(t1ce, cmap='gray')
    mask_np = mask.cpu().numpy()
    plt.imshow(mask_np, cmap='Reds', alpha=0.6)
    plt.title(f"Expert Annotation\nTrue: {Config.CLASSES[true_class]}", fontsize=14)
    plt.axis('off')
    
    # Panel 3: Grad-CAM
    plt.subplot(1, 3, 3)
    plt.imshow(t1ce, cmap='gray')
    gradcam_np = gradcam.cpu().numpy()
    plt.imshow(gradcam_np, cmap='jet', alpha=0.7)
    match = "✓" if pred_class == true_class else "✗"
    plt.title(f"Model Attention\nPred: {Config.CLASSES[pred_class]} {match}", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==================== TRAINING ====================
def train_one_epoch_guided(model, train_loader, cls_criterion, optimizer, epoch, device):
    """Train one epoch with dual loss"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_exp_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Track samples for visualization
    samples_to_viz = []
    
    for batch_idx, (images, labels, masks) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float()
        masks = masks.to(device)
        
        # Forward - Classification
        outputs = model(images).squeeze()
        cls_loss = cls_criterion(outputs, labels)
        
        # Compute Grad-CAM and explanation loss for tumor samples only
        exp_losses = []
        
        # Process samples with tumors (label=1)
        tumor_mask = (labels == 1)
        if tumor_mask.any():
            tumor_images = images[tumor_mask]
            tumor_masks = masks[tumor_mask]
            
            for i in range(tumor_images.size(0)):
                img = tumor_images[i:i+1]
                expert_mask = tumor_masks[i]
                
                # Compute Grad-CAM
                gradcam = model.compute_gradcam(img)
                
                # Explanation loss
                exp_loss = dice_loss(gradcam, expert_mask)
                exp_losses.append(exp_loss)
                
                # Save samples for visualization
                if len(samples_to_viz) < Config.HEATMAPS_PER_EPOCH:
                    samples_to_viz.append({
                        'image': img.squeeze(0).detach(),
                        'mask': expert_mask.detach(),
                        'gradcam': gradcam.detach(),
                        'pred': (torch.sigmoid(outputs[tumor_mask][i]) > 0.5).long().item(),
                        'true': 1
                    })
        
        # Combine losses
        if exp_losses:
            exp_loss = torch.stack(exp_losses).mean()
        else:
            exp_loss = torch.tensor(0.0, device=device)
        
        total_loss = cls_loss + Config.ALPHA * exp_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Metrics
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_exp_loss += exp_loss.item()
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Save heatmaps
    if (epoch + 1) % Config.SAVE_HEATMAPS_EVERY_N_EPOCHS == 0:
        epoch_dir = os.path.join(Config.HEATMAP_DIR, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        for i, sample in enumerate(samples_to_viz[:Config.HEATMAPS_PER_EPOCH]):
            save_path = os.path.join(epoch_dir, f'sample_{i+1}.png')
            save_heatmap_comparison(
                sample['image'], sample['mask'], sample['gradcam'],
                sample['pred'], sample['true'], save_path
            )
    
    epoch_loss = running_loss / len(train_loader)
    epoch_cls_loss = running_cls_loss / len(train_loader)
    epoch_exp_loss = running_exp_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_cls_loss, epoch_exp_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc, all_preds, all_labels, all_probs

def train_guided_model(model, train_loader, val_loader, num_epochs, device):
    """Train with dual loss (classification + explanation)"""
    
    cls_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_exp_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_weights = None
    best_epoch = 0
    
    print("\n" + "="*70)
    print("EXPERT-GUIDED TRAINING START")
    print("="*70)
    print(f"Dual Loss: L_total = L_cls + {Config.ALPHA} * L_exp")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train with dual loss
        train_loss, train_cls_loss, train_exp_loss, train_acc = train_one_epoch_guided(
            model, train_loader, cls_criterion, optimizer, epoch, device
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, cls_criterion, device
        )
        
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_exp_loss'].append(train_exp_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train - Total: {train_loss:.4f}, CLS: {train_cls_loss:.4f}, "
              f"EXP: {train_exp_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_weights = model.state_dict().copy()
            print(f"✓ New best model! (Val Acc: {val_acc:.4f})")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch+1})")
    
    # Restore best weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return history, best_val_acc, best_epoch

# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, device):
    """Comprehensive evaluation with Dice alignment"""
    criterion = nn.BCEWithLogitsLoss()
    
    test_loss, test_acc, all_preds, all_labels, all_probs = validate(
        model, test_loader, criterion, device
    )
    
    # Classification metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Compute Dice alignment for tumor samples
    print("\nComputing attention alignment scores...")
    dice_scores = []
    
    model.eval()
    with torch.no_grad():
        for images, labels, masks in test_loader:
            tumor_mask = (labels == 1)
            if not tumor_mask.any():
                continue
            
            tumor_images = images[tumor_mask].to(device)
            tumor_masks = masks[tumor_mask].to(device)
            
            for i in range(tumor_images.size(0)):
                img = tumor_images[i:i+1]
                expert_mask = tumor_masks[i]
                
                gradcam = model.compute_gradcam(img)
                dice_score = 1 - dice_loss(gradcam, expert_mask).item()
                dice_scores.append(dice_score)
    
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        dice_above_05 = np.sum(np.array(dice_scores) > 0.5) / len(dice_scores) * 100
        dice_above_07 = np.sum(np.array(dice_scores) > 0.7) / len(dice_scores) * 100
    else:
        mean_dice = std_dice = dice_above_05 = dice_above_07 = 0.0
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print(f"Classification Metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    
    print(f"\nAttention Alignment Metrics:")
    print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  Dice > 0.5: {dice_above_05:.1f}%")
    print(f"  Dice > 0.7: {dice_above_07:.1f}%")
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            No Tumor  Tumor")
    print(f"Actual No   {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"       Yes  {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    metrics = {
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'dice_above_05': dice_above_05,
        'dice_above_07': dice_above_07,
        'cm': cm,
        'labels': all_labels,
        'probs': all_probs
    }
    
    return metrics

# ==================== PLOTTING ====================
def plot_training_curves(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train Total Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component Losses
    axes[0, 1].plot(history['train_cls_loss'], label='Classification Loss')
    axes[0, 1].plot(history['train_exp_loss'], label=f'Explanation Loss (×{Config.ALPHA})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(history['train_acc'], label='Train Acc')
    axes[1, 0].plot(history['val_acc'], label='Val Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(Config.VIZ_DIR, 'training_curves_guided.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Training curves saved to: {save_path}")
    plt.close()

def plot_roc_curve(labels, probs):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Guided Model (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Expert-Guided Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(Config.VIZ_DIR, 'roc_curve_guided.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 ROC curve saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Guided Model')
    plt.colorbar()
    
    tick_marks = np.arange(len(Config.CLASSES))
    plt.xticks(tick_marks, Config.CLASSES, rotation=45)
    plt.yticks(tick_marks, Config.CLASSES)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(Config.VIZ_DIR, 'confusion_matrix_guided.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()

# ==================== MAIN ====================
def main():
    print("="*70)
    print("EXPERT-GUIDED BINARY CLASSIFICATION")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Dual Loss: L_total = L_cls + {Config.ALPHA} * L_exp")
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    full_dataset = BinaryBraTSDataset(Config.DATA_DIR)
    
    # Split data
    train_indices, val_indices, test_indices = create_patient_level_splits(full_dataset)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = BinaryClassifierWithGradCAM(pretrained=Config.PRETRAINED)
    model = model.to(Config.DEVICE)
    
    print(f"Model: ResNet50 with Grad-CAM")
    print(f"Training: Dual Loss (BCE + Explanation Alignment)")
    
    # Train
    history, best_val_acc, best_epoch = train_guided_model(
        model, train_loader, val_loader, Config.NUM_EPOCHS, Config.DEVICE
    )
    
    # Plot training
    plot_training_curves(history)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING GUIDED MODEL")
    print("="*70)
    
    metrics = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Visualizations
    plot_roc_curve(metrics['labels'], metrics['probs'])
    plot_confusion_matrix(metrics['cm'])
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'test_acc': metrics['test_acc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'mean_dice': metrics['mean_dice'],
        'std_dice': metrics['std_dice'],
        'dice_above_05': metrics['dice_above_05'],
        'dice_above_07': metrics['dice_above_07'],
        'best_val_acc': best_val_acc,
        'alpha': Config.ALPHA
    }])
    
    metrics_path = os.path.join(Config.LOG_DIR, 'guided_test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save final model
    final_path = os.path.join(Config.MODEL_DIR, 'final_guided_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'test_metrics': metrics,
        'config': {
            'alpha': Config.ALPHA,
            'modalities': Config.USE_MODALITIES,
            'tumor_threshold': Config.TUMOR_THRESHOLD
        }
    }, final_path)
    
    print("\n" + "="*70)
    print("✅ EXPERT-GUIDED TRAINING COMPLETE!")
    print("="*70)
    print(f"Test Accuracy: {metrics['test_acc']:.4f}")
    print(f"Mean Dice Alignment: {metrics['mean_dice']:.4f}")
    print(f"📁 Results: {Config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()