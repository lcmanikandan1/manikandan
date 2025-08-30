import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math
import os
import cv2
import glob
from torchvision.models import vgg19, VGG19_Weights
import piq
from torchvision import transforms as T
import torch.nn.functional as F

class VideoCompressionFramework(nn.Module):
    def __init__(self, latent_dim=64):
        super(VideoCompressionFramework, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.mu_layer = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        self.sigma_layer = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        sigma = torch.abs(self.sigma_layer(latent)) + 1e-6
        recon = self.decoder(latent)

        # Fix temporal mismatch
        if recon.shape[2] > x.shape[2]:
            recon = recon[:, :, :x.shape[2], :, :]
        elif recon.shape[2] < x.shape[2]:
            pad_frames = x.shape[2] - recon.shape[2]
            recon = F.pad(recon, (0, 0, 0, 0, 0, pad_frames))

        return recon, mu, sigma, latent

class VideoFrameDataset(Dataset):
    def __init__(self, root_folder, frames_per_clip=3, transform=None):
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []

        video_files = glob.glob(os.path.join(root_folder, "**", "*.mp4"), recursive=True)
        image_files = glob.glob(os.path.join(root_folder, "**", "*.png"), recursive=True) \
                     + glob.glob(os.path.join(root_folder, "**", "*.jpg"), recursive=True)

        for vf in video_files:
            cap = cv2.VideoCapture(vf)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for start in range(0, total_frames - frames_per_clip + 1, frames_per_clip):
                self.samples.append((vf, start, "video"))

        if len(image_files) >= frames_per_clip:
            for i in range(0, len(image_files) - frames_per_clip + 1, frames_per_clip):
                self.samples.append((image_files[i:i+frames_per_clip], None, "image"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, start, ftype = self.samples[idx]
        frames = []

        if ftype == "video":
            cap = cv2.VideoCapture(item)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(self.frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            cap.release()
        elif ftype == "image":
            for img_path in item:
                frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, video_tensor

class RateDistortionLoss(nn.Module):
    def __init__(self, lambda_rd=0.01):
        super(RateDistortionLoss, self).__init__()
        self.lambda_rd = lambda_rd
        self.mse = nn.MSELoss()

    def forward(self, recon, target, bitrate):
        mse_loss = self.mse(recon, target)
        rd_loss = bitrate + self.lambda_rd * mse_loss
        return rd_loss, mse_loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, recon, target):
        recon_vgg = self.vgg(recon)
        target_vgg = self.vgg(target)
        return torch.mean((recon_vgg - target_vgg) ** 2)

def train_model(model, train_loader, optimizer, scheduler, epochs=3, device="cuda"):
    rate_distortion_loss = RateDistortionLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for videos, targets in train_loader:
            videos, targets = videos.to(device), targets.to(device)
            recon, mu, sigma, _ = model(videos.permute(0, 2, 1, 3, 4))
            bitrate = torch.mean(torch.log2(1 + sigma))
            rd_loss, mse_loss = rate_distortion_loss(recon, targets.permute(0, 2, 1, 3, 4), bitrate)
            perc_loss = perceptual_loss(
                recon.view(-1, 3, recon.shape[-2], recon.shape[-1]),
                targets.view(-1, 3, targets.shape[-2], targets.shape[-1])
            )
            loss = rd_loss + 0.1 * perc_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {total_loss/len(train_loader):.4f}")

def test_model(model, test_loader, device="cuda"):
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for videos, targets in test_loader:
            videos, targets = videos.to(device), targets.to(device)
            recon, _, _, _ = model(videos.permute(0, 2, 1, 3, 4))

            mse = nn.MSELoss()(recon, targets.permute(0, 2, 1, 3, 4)).item()
            psnr = 10 * math.log10(1 / mse)

            recon_frames = recon.permute(0, 2, 1, 3, 4)  
            target_frames = targets.permute(0, 2, 1, 3, 4)
            ssim_total = 0
            count = 0
            for t in range(recon_frames.shape[1]):
                ssim_val = piq.ssim(
                    recon_frames[:, t, :, :, :],
                    target_frames[:, t, :, :, :],
                    data_range=1.
                ).item()
                ssim_total += ssim_val
                count += 1
            ssim = ssim_total / count

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    print(f"Test PSNR: {sum(psnr_scores)/len(psnr_scores):.4f} dB")
    print(f"Test SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoCompressionFramework().to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),  
        T.ToTensor()
    ])

    dataset_path = r"path_of_the_downloaded_dataset"  
    full_dataset = VideoFrameDataset(root_folder=dataset_path, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    print("Starting training...")
    train_model(model, train_loader, optimizer, scheduler, epochs=3, device=device)

    print("Testing...")
    test_model(model, test_loader, device=device)

