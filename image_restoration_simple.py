# ============================================================
#  IMAGE RESTORATION — EDGE RECOVERY (Final Technical Version)
#  Student : Aayush Mishra | USN : 1RN23EC003
#  Tools   : OpenCV, NumPy, Scikit-Image, Matplotlib
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # ESSENTIAL for folder management
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import unsharp_mask

# ─────────────────────────────────
# STEP 0: ENSURE RESULTS FOLDER
# ─────────────────────────────────
# If you run the report asap, we must handle the folder issue automatically.
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────
# STEP 1: LOAD IMAGE
# ─────────────────────────────────
img_path = "T3700580.jpg"
img = cv2.imread(img_path)

# If image not found, create a simple test image
if img is None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (120, 120), (255, 255, 255), -1)
    cv2.circle(img,   (180, 80), 50,         (180, 180, 180), -1)
    pts = np.array([[60,200],[130,140],[200,200]], np.int32)
    cv2.fillPoly(img, [pts], (100, 150, 220))
    print(f"[!] {img_path} not found — using synthetic test image.")
else:
    print(f"[v] Loaded image: {img_path}, shape = {img.shape}")

# Resize to standard size - CRITICAL: Must be 1024x1024 for PCB complexity
img = cv2.resize(img, (1024, 1024))
print("Step 1 done: Image standardisation done.")

# ─────────────────────────────────
# STEP 2: PRE-PROCESSING (CLAHE)
# ─────────────────────────────────
# CLAHE improves contrast before we do any restoration
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
preprocessed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
print("Step 2 done: CLAHE contrast applied.")

# ─────────────────────────────────
# STEP 3: ADD DEGRADATION (Blur + Noise)
# ─────────────────────────────────
# Blur mimics camera defocus
blurred = cv2.GaussianBlur(preprocessed, (3, 3), 0.8)

# Noise mimics sensor noise
noise   = np.random.normal(0, 12, blurred.shape).astype(np.float32)
degraded = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
print("Step 3 done: Degradation model applied.")

# ─────────────────────────────────
# STEP 4: DENOISE — Non-Local Means
# ─────────────────────────────────
# NLM compares similar patches across the image to remove noise while keeping edges sharp
img_float = img_as_float(degraded)
sigma_est = estimate_sigma(img_float, channel_axis=-1, average_sigmas=True)
h_value   = 0.6 * sigma_est # Pro-tuner value for PCB traces

nlm_restored = denoise_nl_means(img_float,
                                h=h_value,
                                fast_mode=True,
                                patch_size=3,
                                patch_distance=5,
                                channel_axis=-1)
nlm_restored = img_as_ubyte(np.clip(nlm_restored, 0, 1))
print(f"Step 4 done: NLM denoising complete.")

# ─────────────────────────────────
# STEP 5: DENOISE — Bilateral Filter (for comparison)
# ─────────────────────────────────
bilateral = cv2.bilateralFilter(degraded, d=9, sigmaColor=75, sigmaSpace=75)
print("Step 5 done: Bilateral filter complete.")

# ─────────────────────────────────
# STEP 6: EDGE DETECTION
# ─────────────────────────────────
# Perform edge detection on the CLEANEST image (NLM output)
gray = cv2.cvtColor(nlm_restored, cv2.COLOR_BGR2GRAY)

canny     = cv2.Canny(gray, 50, 150)

sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel     = cv2.convertScaleAbs(np.sqrt(sobelx**2 + sobely**2))

laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))

print("Step 6 done: Canny, Sobel, Laplacian edges computed.")

# ─────────────────────────────────
# STEP 7: EDGE RESTORATION (Selective Masking)
# ─────────────────────────────────
# Stage A: Unsharp Masking — amplifies high-frequency edge info
sharp_float = img_as_float(nlm_restored)
sharpened   = unsharp_mask(sharp_float, radius=1.5, amount=2.5, channel_axis=-1)

# Stage B: Create a smooth stencil from Canny Edges
# Normalise and blur mask to prevent artifacts on traces
mask     = canny.astype(np.float64) / 255.0
mask     = cv2.GaussianBlur(mask, (3, 3), 0)
mask_3ch = np.stack([mask]*3, axis=-1)

# Stage C: Selective Blending with Hardened Clipping
alpha    = 0.5 # Slightly lower alpha for 1024x1024 to prevent overflow
final_f  = (sharpened * mask_3ch * alpha) + (sharp_float * (1.0 - mask_3ch * alpha))

# CRITICAL FIX: Explicitly clip to [0, 1] before converting back to pixels
final_f  = np.clip(final_f, 0.0, 1.0) 
final    = img_as_ubyte(final_f)
print("Step 7 done: Selective trace sharpening complete.")

# ─────────────────────────────────
# STEP 8: QUALITY METRICS (PSNR & SSIM)
# ─────────────────────────────────
def get_metrics(orig, result):
    o = img_as_float(cv2.cvtColor(orig,   cv2.COLOR_BGR2GRAY))
    r = img_as_float(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    return round(psnr(o, r, data_range=1.0), 2), \
           round(ssim(o, r, data_range=1.0), 4)

p1, s1 = get_metrics(preprocessed, degraded)
p2, s2 = get_metrics(preprocessed, nlm_restored)
p3, s3 = get_metrics(preprocessed, bilateral)
p4, s4 = get_metrics(preprocessed, final)

print("\n" + "="*48)
print(f"  {'Stage':<22} {'PSNR (dB)':>10} {'SSIM':>10}")
print("="*48)
print(f"  {'Degraded':<22} {p1:>10} {s1:>10}")
print(f"  {'NLM Restored':<22} {p2:>10} {s2:>10}")
print(f"  {'Bilateral Restored':<22} {p3:>10} {s3:>10}")
print(f"  {'Edge Restored':<22} {p4:>10} {s4:>10}")
print("="*48)

# ─────────────────────────────────
# STEP 9: VISUALISE ALL RESULTS
# ─────────────────────────────────
print("\n[STEP 9] Generating final report figures …")
fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.patch.set_facecolor("#0f0f0f") # Professional dark presentation
fig.suptitle("Image Restoration Pipeline — Edge Recovery\n"
             "Aayush Mishra | 1RN23EC003 | RNSIT (BEC613C)",
             fontsize=13, fontweight="bold", color="white", y=0.99)

panels = [
    # Row 1: Input stages
    (cv2.cvtColor(img, 
                 cv2.COLOR_BGR2RGB), "① Original\n(512x512)", None),
    (cv2.cvtColor(preprocessed, 
                 cv2.COLOR_BGR2RGB), "② Pre-processed\n(CLAHE)", None),
    (cv2.cvtColor(degraded, 
                 cv2.COLOR_BGR2RGB), "③ Degraded\n(Blur σ=0.8, Noise std=12)", None),
    # Row 2: Restoration stages
    (cv2.cvtColor(nlm_restored, 
                 cv2.COLOR_BGR2RGB), "④ NLM Denoised\n(h=0.6*σ_est)", None),
    (cv2.cvtColor(bilateral, 
                 cv2.COLOR_BGR2RGB), "⑤ Bilateral Denoised\n(d=9, c=75, s=75)", None),
    (cv2.cvtColor(final, 
                 cv2.COLOR_BGR2RGB), "⑥ FINAL Output ✦\nSelective Edge Sharpness", None),
    # Row 3: Edge maps
    (canny, "⑦ Canny Edges (Staging)", "hot"),
    (sobel, "⑧ Sobel Edges (Detail)", "hot"),
    (laplacian, "⑨ Laplacian Edges (Detail)", "hot"),
]

for i, (panel, title, cmap) in enumerate(panels):
    ax = axes[i // 3][i % 3]
    ax.imshow(panel, cmap=cmap)
    ax.set_facecolor("#0f0f0f")

    # Highlight the final output
    if "✦" in title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="#0f0f0f",
                     bbox=dict(facecolor="#00e5ff", pad=4,
                               edgecolor="none", boxstyle="round,pad=0.4"))
    else:
        row = i // 3
        colour = ["#ffffff", "#ffffff", "#00e5ff"][row]
        ax.set_title(title, fontsize=9.5, color=colour, fontweight="bold", pad=7)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(pad=1.0)
fig_out = "results/pipeline_result.png"
plt.savefig(fig_out, dpi=160,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[v] Pipeline complete. Figures saved → {fig_out}")
plt.show()