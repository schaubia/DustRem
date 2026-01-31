import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure page
st.set_page_config(
    page_title="Deep Learning Dust Remover",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple U-Net Model
class UNet(nn.Module):
    """
    Simplified U-Net for dust spot detection/removal.
    Can be trained on user examples or use transfer learning.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder with skip connections
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))

def create_training_data_from_marks(image, spots, patch_size=128):
    """Create training data from user-marked spots."""
    height, width = image.shape[:2]
    
    # Create mask from marked spots
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, r in spots:
        cv2.circle(mask, (x, y), r, 255, -1)
    
    # Extract patches
    patches_clean = []
    patches_dusty = []
    masks = []
    
    for spot in spots:
        x, y, r = spot
        
        # Extract patch around spot
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = min(width, x1 + patch_size)
        y2 = min(height, y1 + patch_size)
        
        # Ensure patch is correct size
        if x2 - x1 == patch_size and y2 - y1 == patch_size:
            dusty_patch = image[y1:y2, x1:x2].copy()
            mask_patch = mask[y1:y2, x1:x2].copy()
            
            # Simulate clean version by inpainting
            clean_patch = cv2.inpaint(dusty_patch, mask_patch, 3, cv2.INPAINT_TELEA)
            
            patches_dusty.append(dusty_patch)
            patches_clean.append(clean_patch)
            masks.append(mask_patch)
    
    return patches_dusty, patches_clean, masks

def train_unet_on_examples(model, dusty_patches, clean_patches, masks, device, epochs=50):
    """Train U-Net on user examples."""
    if not dusty_patches:
        return model, []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    losses = []
    model.train()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for dusty, mask in zip(dusty_patches, masks):
            # Convert to tensors
            dusty_tensor = torch.from_numpy(dusty.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            # Forward pass
            optimizer.zero_grad()
            output = model(dusty_tensor)
            
            # Loss
            loss = criterion(output, mask_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dusty_patches)
        losses.append(avg_loss)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    progress_bar.empty()
    status_text.empty()
    
    model.eval()
    return model, losses

def detect_dust_with_unet(model, image, device, threshold=0.5):
    """Use trained U-Net to detect dust spots."""
    try:
        height, width = image.shape[:2]
        
        # Prepare image
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        
        # Ensure dimensions are divisible by 8 for U-Net
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :height, :width]
        
        # Convert to mask
        mask = (output.squeeze().cpu().numpy() * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        return binary_mask
        
    except Exception as e:
        st.error(f"Error in U-Net detection: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)

def remove_dust_spots(image, mask):
    """Remove dust spots using inpainting."""
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    return result

# Streamlit UI
st.title("🤖 Deep Learning Dust Remover (U-Net)")
st.markdown("""
**Train a U-Net neural network to detect and remove YOUR specific dust spots!**

This version uses deep learning instead of traditional computer vision for more accurate detection.
""")

# Initialize session state
if 'training_spots' not in st.session_state:
    st.session_state.training_spots = []
if 'unet_model' not in st.session_state:
    st.session_state.unet_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'cleaned_image' not in st.session_state:
    st.session_state.cleaned_image = None

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    st.sidebar.success(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("💻 Using CPU (slower training)")

# Sidebar
st.sidebar.header("Deep Learning Settings")

mode = st.sidebar.radio(
    "Choose mode",
    ["🧠 Train U-Net Model", "✏️ Manual Removal Only"],
    help="Train neural network or manual marking"
)

if mode == "🧠 Train U-Net Model":
    st.sidebar.markdown("""
    ### How it works:
    1. **Mark dust spots** (3-10 examples)
    2. **Train U-Net** neural network
    3. **AI detects** all similar spots
    4. **Remove** detected spots
    
    U-Net learns spatial patterns, not just colors!
    """)
    
    training_epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="More epochs = better training but slower"
    )
    
    detection_threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower = detect more spots"
    )
else:
    st.sidebar.markdown("### Manual mode")
    brush_size = st.sidebar.slider("Spot Radius", 10, 150, 40)

# File uploader
st.markdown("### Upload Your Image")

upload_method = st.radio(
    "Choose upload method:",
    ["📷 Upload clean image", "🔴 Upload image with red circles already marked"],
    horizontal=True
)

if upload_method == "📷 Upload clean image":
    uploaded_file = st.file_uploader(
        "Upload your image with dust spots",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="clean_upload"
    )
    pre_marked = None
else:
    uploaded_file = st.file_uploader(
        "Upload image with RED circles marking dust spots",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="marked_upload_main"
    )
    pre_marked = uploaded_file

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Auto-detect red circles if pre-marked
    if pre_marked is not None and len(st.session_state.training_spots) == 0:
        with st.spinner("Detecting red markings..."):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 20:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        r = max(int(np.sqrt(area / np.pi)) + 5, 50)
                        st.session_state.training_spots.append((cx, cy, r))
                        detected_count += 1
            
            if detected_count > 0:
                st.success(f"✅ Auto-detected {detected_count} red-marked spots!")
            else:
                st.warning("⚠️ No red markings detected. Mark manually below.")
    
    if mode == "🧠 Train U-Net Model":
        st.header("Step 1: Mark Training Examples")
        
        if pre_marked is None:
            st.markdown("Mark 3-10 dust spots for the neural network to learn from")
        else:
            st.markdown("✅ Using red circles as training examples")
        
        # Show preview
        if pre_marked is None:
            preview = image_rgb.copy()
            for spot in st.session_state.training_spots:
                x, y, r = spot
                cv2.circle(preview, (x, y), r, (255, 0, 0), 2)
                cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
        else:
            preview = image_rgb.copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(preview, width="stretch", caption=f"Training examples: {len(st.session_state.training_spots)}")
        
        with col2:
            st.markdown("### Add Example")
            x_input = st.number_input("X", 0, image.shape[1], 0, key="x_train")
            y_input = st.number_input("Y", 0, image.shape[0], 0, key="y_train")
            r_input = st.number_input("Radius", 10, 200, 50, key="r_train")
            
            if st.button("➕ Add", type="primary"):
                st.session_state.training_spots.append((x_input, y_input, r_input))
                st.rerun()
            
            if st.session_state.training_spots:
                if st.button("🗑️ Clear"):
                    st.session_state.training_spots = []
                    st.session_state.model_trained = False
                    st.rerun()
        
        # Bulk input
        with st.expander("📋 Paste coordinates"):
            coords_text = st.text_area("x,y,radius (one per line)", height=100)
            if st.button("Add All"):
                for line in coords_text.strip().split('\n'):
                    try:
                        x, y, r = map(int, [p.strip() for p in line.split(',')])
                        st.session_state.training_spots.append((x, y, r))
                    except:
                        pass
                st.rerun()
        
        # Training section
        if len(st.session_state.training_spots) >= 3:
            st.markdown("---")
            st.header("Step 2: Train U-Net Neural Network")
            
            if st.button("🧠 Train U-Net Model", type="primary"):
                with st.spinner("Training neural network..."):
                    # Create training data
                    dusty_patches, clean_patches, masks = create_training_data_from_marks(
                        image, st.session_state.training_spots
                    )
                    
                    if dusty_patches:
                        # Initialize model
                        model = UNet(in_channels=3, out_channels=1).to(device)
                        
                        # Train
                        model, losses = train_unet_on_examples(
                            model, dusty_patches, clean_patches, masks,
                            device, epochs=training_epochs
                        )
                        
                        st.session_state.unet_model = model
                        st.session_state.model_trained = True
                        
                        st.success(f"✅ U-Net trained on {len(dusty_patches)} examples!")
                        
                        # Show training curve
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(losses)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Progress')
                        ax.grid(True)
                        st.pyplot(fig)
            
            if st.session_state.model_trained and st.session_state.unet_model:
                if st.button("🔍 Detect Dust with U-Net", type="primary"):
                    with st.spinner("U-Net analyzing image..."):
                        detected_mask = detect_dust_with_unet(
                            st.session_state.unet_model,
                            image,
                            device,
                            threshold=detection_threshold
                        )
                        
                        if cv2.countNonZero(detected_mask) > 0:
                            # Show detected
                            vis = image_rgb.copy()
                            contours, _ = cv2.findContours(detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for contour in contours:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    radius = int(np.sqrt(cv2.contourArea(contour) / np.pi)) + 5
                                    cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 2)
                            
                            st.subheader(f"U-Net Found {len(contours)} Spots!")
                            st.image(vis, width="stretch")
                            
                            with st.expander("View mask"):
                                st.image(detected_mask, width="stretch")
                            
                            if st.button("✅ Remove Detected Spots"):
                                cleaned = remove_dust_spots(image, detected_mask)
                                cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Original**")
                                    st.image(image_rgb, width="stretch")
                                with col2:
                                    st.markdown("**Cleaned**")
                                    st.image(cleaned_rgb, width="stretch")
                                
                                st.success("✅ Cleaned!")
                                
                                pil_image = Image.fromarray(cleaned_rgb)
                                buf = io.BytesIO()
                                pil_image.save(buf, format="PNG")
                                
                                st.download_button(
                                    "⬇️ Download",
                                    data=buf.getvalue(),
                                    file_name="unet_cleaned.png",
                                    mime="image/png"
                                )
                        else:
                            st.info("No spots detected. Try lowering threshold.")
                
                # Apply to another image
                st.markdown("---")
                st.subheader("🔄 Apply to Another Image")
                
                other_file = st.file_uploader(
                    "Upload another image",
                    type=["jpg", "jpeg", "png"],
                    key="other"
                )
                
                if other_file:
                    other_bytes = np.asarray(bytearray(other_file.read()), dtype=np.uint8)
                    other_image = cv2.imdecode(other_bytes, cv2.IMREAD_COLOR)
                    other_rgb = cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB)
                    
                    st.image(other_rgb, width="stretch")
                    
                    if st.button("🔍 Detect on This Image"):
                        other_mask = detect_dust_with_unet(
                            st.session_state.unet_model,
                            other_image,
                            device,
                            threshold=detection_threshold
                        )
                        
                        if cv2.countNonZero(other_mask) > 0:
                            other_cleaned = remove_dust_spots(other_image, other_mask)
                            other_cleaned_rgb = cv2.cvtColor(other_cleaned, cv2.COLOR_BGR2RGB)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(other_rgb, width="stretch")
                            with col2:
                                st.image(other_cleaned_rgb, width="stretch")
                            
                            pil_other = Image.fromarray(other_cleaned_rgb)
                            buf_other = io.BytesIO()
                            pil_other.save(buf_other, format="PNG")
                            
                            st.download_button(
                                "⬇️ Download",
                                data=buf_other.getvalue(),
                                file_name="other_cleaned.png",
                                mime="image/png",
                                key="download_other"
                            )
        else:
            st.info("Mark at least 3 spots to train U-Net")
    
    else:
        # Manual mode (same as before)
        st.header("Manual Removal")
        # ... (manual mode code would go here)
        st.info("Manual mode: Mark spots and remove directly")

else:
    st.info("👆 Upload an image to start")
    
    st.markdown("---")
    st.subheader("🧠 Why U-Net?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Traditional CV
        - Rule-based algorithms
        - Fixed feature detection
        - Can't learn patterns
        - Limited adaptability
        """)
    
    with col2:
        st.markdown("""
        ### U-Net Deep Learning
        - Learns from examples
        - Captures spatial patterns
        - Adapts to your dust type
        - State-of-the-art accuracy
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Deep Learning Dust Remover | U-Net Architecture</div>", unsafe_allow_html=True)
