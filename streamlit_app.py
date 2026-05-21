import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Learning Dust Remover",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="expanded",
)


# ── U-Net model ───────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.bottleneck = self._block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def _block(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b  = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.upconv3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


# ── Helper functions ──────────────────────────────────────────────────────────
def create_training_data(image, spots, patch_size=128):
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, r in spots:
        cv2.circle(full_mask, (x, y), r, 255, -1)

    dusty_patches, mask_patches = [], []
    for x, y, r in spots:
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        if x2 > w or y2 > h:
            continue
        dusty_patches.append(image[y1:y2, x1:x2].copy())
        mask_patches.append(full_mask[y1:y2, x1:x2].copy())
    return dusty_patches, mask_patches


def train_unet(model, dusty_patches, mask_patches, device, epochs, freeze_encoder=False):
    if freeze_encoder:
        for name, p in model.named_parameters():
            if "enc1" in name or "enc2" in name:
                p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCELoss()
    model.train()
    losses = []
    bar = st.progress(0)
    txt = st.empty()

    for epoch in range(epochs):
        total = 0.0
        for dusty, mask in zip(dusty_patches, mask_patches):
            inp = torch.from_numpy(dusty.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            tgt = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            optimizer.zero_grad()
            loss = criterion(model(inp), tgt)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(len(dusty_patches), 1)
        losses.append(avg)
        bar.progress((epoch + 1) / epochs)
        txt.text(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    bar.empty(); txt.empty()
    model.eval()
    return model, losses


def detect_with_unet(model, image, device, threshold=0.5):
    h, w = image.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        out = model(t)
    out = out[:, :, :h, :w]
    mask = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
    return binary


def inpaint(image, mask, radius=7):
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)


def detect_red_circles(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(m1, m2)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        r  = max(int(np.sqrt(area / np.pi)) + 5, 20)
        spots.append((cx, cy, r))
    return spots


def overlay_spots(image_rgb, spots, color=(255, 50, 50)):
    vis = image_rgb.copy()
    for x, y, r in spots:
        cv2.circle(vis, (x, y), r, color, 2)
        cv2.circle(vis, (x, y), 3, color, -1)
    return vis


# ── Session-state initialisation ──────────────────────────────────────────────
def _init():
    defaults = dict(
        spots=[],
        model=None,
        trained=False,
        detected_mask=None,
        cleaned_rgb=None,
        image_bgr=None,
        image_rgb=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

if device.type == "cuda":
    st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("💻 CPU mode (training may be slow)")

mode = st.sidebar.radio(
    "Mode",
    ["🧠 Train U-Net", "✏️ Manual Removal"],
    help="Train the neural network, or mark and remove spots manually",
)

# These variables are always defined to avoid NameErrors
use_tl        = False
epochs        = 30
det_threshold = 0.5
brush_size    = 40

if mode == "🧠 Train U-Net":
    use_tl = st.sidebar.checkbox(
        "Use Transfer Learning (freeze encoder)",
        value=False,
        help="Faster convergence, better results with few examples",
    )
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 20 if use_tl else 30, 10)
    det_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                                      help="Lower → detect more spots")
    st.sidebar.markdown("""
    **Workflow**
    1. Upload image → mark 3–10 spots
    2. Train U-Net
    3. Detect all similar spots
    4. Remove & download
    """)
else:
    brush_size = st.sidebar.slider("Spot Radius", 10, 200, 50)
    st.sidebar.markdown("""
    **Workflow**
    1. Upload image
    2. Enter spot coordinates
    3. Remove & download
    """)


# ── Main title ─────────────────────────────────────────────────────────────────
st.title("📸 Deep Learning Dust Spot Remover")
st.caption("U-Net neural network learns YOUR dust patterns for precise removal.")


# ── Upload ─────────────────────────────────────────────────────────────────────
st.subheader("1 · Upload Image")
upload_method = st.radio(
    "Upload method:",
    ["📷 Clean image (mark spots manually)", "🔴 Image already marked with red circles"],
    horizontal=True,
)
pre_marked = upload_method.startswith("🔴")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    key="uploader",
)

if uploaded:
    raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Only reset when a new file arrives
    if st.session_state.image_bgr is None or bgr.shape != st.session_state.image_bgr.shape:
        st.session_state.image_bgr    = bgr
        st.session_state.image_rgb    = rgb
        st.session_state.spots        = []
        st.session_state.trained      = False
        st.session_state.model        = None
        st.session_state.detected_mask = None
        st.session_state.cleaned_rgb  = None

    bgr = st.session_state.image_bgr
    rgb = st.session_state.image_rgb

    # Auto-detect red circles on first load of a pre-marked image
    if pre_marked and len(st.session_state.spots) == 0:
        with st.spinner("Detecting red markings…"):
            found = detect_red_circles(bgr)
        if found:
            st.session_state.spots = found
            st.success(f"✅ Auto-detected {len(found)} red-marked spots.")
        else:
            st.warning("⚠️ No red markings found – add spots manually below.")


# ── Content (only when image is loaded) ────────────────────────────────────────
if st.session_state.image_bgr is not None:
    bgr = st.session_state.image_bgr
    rgb = st.session_state.image_rgb

    # ── SPOT MARKING ───────────────────────────────────────────────────────────
    st.subheader("2 · Mark Dust Spots")
    col_img, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("**Add a spot**")
        xi = st.number_input("X", 0, bgr.shape[1] - 1, 0, key="xi")
        yi = st.number_input("Y", 0, bgr.shape[0] - 1, 0, key="yi")
        ri = st.number_input("Radius", 5, 300, brush_size, key="ri")

        if st.button("➕ Add spot", type="primary"):
            st.session_state.spots.append((int(xi), int(yi), int(ri)))
            st.rerun()

        if st.session_state.spots:
            if st.button("🗑️ Clear all spots"):
                st.session_state.spots        = []
                st.session_state.trained      = False
                st.session_state.model        = None
                st.session_state.detected_mask = None
                st.session_state.cleaned_rgb  = None
                st.rerun()

        with st.expander("📋 Paste multiple spots"):
            bulk = st.text_area("x,y,radius – one per line", height=120)
            if st.button("Add all"):
                for line in bulk.strip().splitlines():
                    try:
                        x, y, r = map(int, [p.strip() for p in line.split(",")])
                        st.session_state.spots.append((x, y, r))
                    except Exception:
                        pass
                st.rerun()

        st.caption(f"**{len(st.session_state.spots)} spot(s) marked**")

    with col_img:
        preview = overlay_spots(rgb, st.session_state.spots)
        st.image(preview, use_container_width=True,
                 caption="Marked spots shown in red")

    # ── MODE: MANUAL ───────────────────────────────────────────────────────────
    if mode == "✏️ Manual Removal":
        st.subheader("3 · Remove Marked Spots")
        if not st.session_state.spots:
            st.info("Mark at least one spot above, then click Remove.")
        else:
            if st.button("🧹 Remove marked spots", type="primary"):
                mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                for x, y, r in st.session_state.spots:
                    cv2.circle(mask, (x, y), r, 255, -1)
                cleaned_bgr = inpaint(bgr, mask)
                st.session_state.cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        if st.session_state.cleaned_rgb is not None:
            st.success("✅ Spots removed!")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(rgb, use_container_width=True)
            with c2:
                st.markdown("**Cleaned**")
                st.image(st.session_state.cleaned_rgb, use_container_width=True)

            buf = io.BytesIO()
            Image.fromarray(st.session_state.cleaned_rgb).save(buf, format="PNG")
            st.download_button("⬇️ Download cleaned image", buf.getvalue(),
                               "cleaned.png", "image/png")

    # ── MODE: U-NET ────────────────────────────────────────────────────────────
    else:
        # ── TRAIN ──────────────────────────────────────────────────────────────
        st.subheader("3 · Train U-Net")
        if len(st.session_state.spots) < 3:
            st.info("Mark at least **3 spots** above to enable training.")
        else:
            if st.button("🧠 Train U-Net model", type="primary"):
                dusty, masks = create_training_data(bgr, st.session_state.spots)
                if not dusty:
                    st.error("No valid patches extracted – try larger spot radii or spots not too close to image edges.")
                else:
                    model = UNet().to(device)
                    tl_note = " (encoder frozen)" if use_tl else ""
                    st.info(f"Training on {len(dusty)} patch(es){tl_note}…")
                    model, losses = train_unet(model, dusty, masks, device, epochs, freeze_encoder=use_tl)
                    st.session_state.model   = model
                    st.session_state.trained = True
                    st.session_state.detected_mask = None  # reset previous detection
                    st.session_state.cleaned_rgb   = None

                    st.success(f"✅ Trained {epochs} epoch(s) on {len(dusty)} patch(es){tl_note}!")

                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    ax.plot(losses, color="#4c9be8")
                    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
                    ax.set_title("Training Loss" + tl_note); ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

        # ── DETECT ─────────────────────────────────────────────────────────────
        if st.session_state.trained and st.session_state.model:
            st.subheader("4 · Detect Dust Spots")
            if st.button("🔍 Run U-Net detection", type="primary"):
                with st.spinner("Running U-Net…"):
                    mask = detect_with_unet(st.session_state.model, bgr, device, det_threshold)
                st.session_state.detected_mask = mask
                st.session_state.cleaned_rgb   = None  # reset previous result

            if st.session_state.detected_mask is not None:
                dmask = st.session_state.detected_mask
                n_contours = len(cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

                if cv2.countNonZero(dmask) == 0:
                    st.warning("No spots detected – try lowering the Detection Threshold.")
                else:
                    # Visualise detections
                    vis = rgb.copy()
                    contours, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cr = int(np.sqrt(cv2.contourArea(c) / np.pi)) + 3
                            cv2.circle(vis, (cx, cy), cr, (0, 230, 80), 2)
                    st.success(f"Found **{n_contours}** spot(s) – shown in green below.")
                    st.image(vis, use_container_width=True, caption="Detected spots")

                    with st.expander("View raw detection mask"):
                        st.image(dmask, use_container_width=True)

                    # ── REMOVE ─────────────────────────────────────────────────
                    st.subheader("5 · Remove & Download")
                    if st.button("✅ Remove detected spots", type="primary"):
                        cleaned_bgr = inpaint(bgr, dmask)
                        st.session_state.cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

                    if st.session_state.cleaned_rgb is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Original**")
                            st.image(rgb, use_container_width=True)
                        with c2:
                            st.markdown("**Cleaned**")
                            st.image(st.session_state.cleaned_rgb, use_container_width=True)

                        buf = io.BytesIO()
                        Image.fromarray(st.session_state.cleaned_rgb).save(buf, format="PNG")
                        st.download_button("⬇️ Download cleaned image", buf.getvalue(),
                                           "unet_cleaned.png", "image/png")

            # ── APPLY TO ANOTHER IMAGE ──────────────────────────────────────────
            st.markdown("---")
            st.subheader("Apply trained model to another image")
            other = st.file_uploader("Upload another image", type=["jpg","jpeg","png","bmp","tiff"],
                                     key="other_img")
            if other:
                ob  = np.asarray(bytearray(other.read()), dtype=np.uint8)
                obgr = cv2.imdecode(ob, cv2.IMREAD_COLOR)
                orgb = cv2.cvtColor(obgr, cv2.COLOR_BGR2RGB)
                st.image(orgb, use_container_width=True, caption="Other image")
                if st.button("🔍 Detect & remove on this image"):
                    with st.spinner("Detecting…"):
                        omask = detect_with_unet(st.session_state.model, obgr, device, det_threshold)
                    if cv2.countNonZero(omask) > 0:
                        ocleaned = cv2.cvtColor(inpaint(obgr, omask), cv2.COLOR_BGR2RGB)
                        c1, c2 = st.columns(2)
                        with c1: st.image(orgb,     use_container_width=True, caption="Before")
                        with c2: st.image(ocleaned, use_container_width=True, caption="After")
                        obuf = io.BytesIO()
                        Image.fromarray(ocleaned).save(obuf, format="PNG")
                        st.download_button("⬇️ Download", obuf.getvalue(),
                                           "other_cleaned.png", "image/png", key="dl_other")
                    else:
                        st.info("No spots detected on this image.")

else:
    # Welcome screen
    st.info("👆 Upload an image above to get started.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **🧠 U-Net mode**
        - Mark 3–10 example spots
        - Neural network trains on your examples
        - Detects all similar spots automatically
        - State-of-the-art inpainting
        """)
    with c2:
        st.markdown("""
        **✏️ Manual mode**
        - Enter coordinates of each spot
        - Instant removal, no training needed
        - Great for a small number of known spots
        """)

st.markdown("---")
st.caption("Deep Learning Dust Remover · U-Net Architecture · Built with PyTorch & Streamlit")
