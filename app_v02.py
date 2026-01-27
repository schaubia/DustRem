import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def create_training_patches(image, mask, patch_size=64):
    """Create training patches from user-marked spots."""
    height, width = image.shape[:2]
    positive_patches = []
    negative_patches = []
    
    # Get positive samples (dust spots)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Extract patch around dust spot
            x1 = max(0, cx - patch_size // 2)
            y1 = max(0, cy - patch_size // 2)
            x2 = min(width, x1 + patch_size)
            y2 = min(height, y1 + patch_size)
            
            if x2 - x1 == patch_size and y2 - y1 == patch_size:
                patch = image[y1:y2, x1:x2]
                positive_patches.append(patch)
    
    # Get negative samples (clean areas)
    for _ in range(len(positive_patches) * 2):
        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)
        
        # Check if this area overlaps with dust spots
        mask_patch = mask[y:y+patch_size, x:x+patch_size]
        if np.sum(mask_patch) == 0:  # No dust in this patch
            patch = image[y:y+patch_size, x:x+patch_size]
            negative_patches.append(patch)
    
    return positive_patches, negative_patches

def detect_dust_ml_simple(image, positive_examples, negative_examples, sensitivity=0.5):
    """
    Simple ML-based detection using color/texture statistics from examples.
    This is a lightweight approach that learns from your marked examples.
    """
    if not positive_examples:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert to LAB color space for better color analysis
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract features from positive examples (dust spots)
    dust_features = []
    for patch in positive_examples:
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        
        # Calculate statistics
        mean_intensity = np.mean(patch_gray)
        std_intensity = np.std(patch_gray)
        mean_l = np.mean(patch_lab[:,:,0])
        mean_a = np.mean(patch_lab[:,:,1])
        mean_b = np.mean(patch_lab[:,:,2])
        
        dust_features.append({
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'mean_l': mean_l,
            'mean_a': mean_a,
            'mean_b': mean_b
        })
    
    # Calculate average dust characteristics
    avg_intensity = np.mean([f['mean_intensity'] for f in dust_features])
    avg_std = np.mean([f['std_intensity'] for f in dust_features])
    avg_l = np.mean([f['mean_l'] for f in dust_features])
    avg_a = np.mean([f['mean_a'] for f in dust_features])
    avg_b = np.mean([f['mean_b'] for f in dust_features])
    
    # Create sliding window detection
    window_size = 32
    stride = 8
    height, width = image.shape[:2]
    
    detection_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            # Extract window
            window = image_gray[y:y+window_size, x:x+window_size]
            window_lab = image_lab[y:y+window_size, x:x+window_size]
            
            # Calculate similarity to dust
            win_intensity = np.mean(window)
            win_std = np.std(window)
            win_l = np.mean(window_lab[:,:,0])
            win_a = np.mean(window_lab[:,:,1])
            win_b = np.mean(window_lab[:,:,2])
            
            # Calculate distance from dust characteristics
            intensity_diff = abs(win_intensity - avg_intensity)
            std_diff = abs(win_std - avg_std)
            l_diff = abs(win_l - avg_l)
            a_diff = abs(win_a - avg_a)
            b_diff = abs(win_b - avg_b)
            
            # Composite score (lower is more similar to dust)
            score = (intensity_diff/255 + std_diff/255 + 
                    l_diff/255 + a_diff/255 + b_diff/255) / 5
            
            # Mark areas similar to dust
            if score < sensitivity:
                detection_map[y:y+window_size, x:x+window_size] = 1
    
    # Convert to binary mask
    detection_map_uint8 = (detection_map * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(detection_map_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary_mask

def remove_dust_spots(image, mask):
    """Remove dust spots from image using inpainting."""
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    return result

# Streamlit UI
st.set_page_config(page_title="ML Dust Spot Remover", layout="wide")

st.title("🤖 Machine Learning Dust Spot Remover")
st.markdown("""
**Train the AI to recognize YOUR specific dust spots!**

This app learns from examples you provide and finds similar spots automatically.
""")

# Initialize session state
if 'training_spots' not in st.session_state:
    st.session_state.training_spots = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'positive_patches' not in st.session_state:
    st.session_state.positive_patches = []
if 'negative_patches' not in st.session_state:
    st.session_state.negative_patches = []
if 'cleaned_image' not in st.session_state:
    st.session_state.cleaned_image = None

# Sidebar
st.sidebar.header("🎓 Training Mode")

mode = st.sidebar.radio(
    "Choose your approach",
    ["📝 Mark Examples & Train", "🔍 Manual Removal Only"],
    help="Train AI on examples or manually mark all spots"
)

if mode == "📝 Mark Examples & Train":
    st.sidebar.markdown("""
    ### How it works:
    1. **Mark a few dust spots** (3-5 examples)
    2. **Click "Train AI"** to learn from examples
    3. **AI finds similar spots** automatically
    4. **Review and remove** detected spots
    
    The AI learns the color, brightness, and texture of your dust spots!
    """)
    
    ml_sensitivity = st.sidebar.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower = find more similar spots"
    )
    
else:
    st.sidebar.markdown("""
    ### Manual mode:
    Mark all dust spots manually and remove them directly.
    """)
    
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
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # If pre-marked image was uploaded, auto-detect red circles
    if pre_marked is not None and len(st.session_state.training_spots) == 0:
        with st.spinner("Detecting red markings..."):
            # Detect red areas
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
                st.success(f"✅ Auto-detected {detected_count} red-marked spots! These are now loaded as training examples.")
                st.info("You can add more examples below, or proceed to train the AI.")
            else:
                st.warning("⚠️ No red markings detected. Make sure you used bright red color (RGB: 255, 0, 0). You can still add examples manually below.")
    
    
    if mode == "📝 Mark Examples & Train":
        st.header("Step 1: Mark Example Dust Spots")
        
        if pre_marked is None:
            st.markdown("Mark 3-5 dust spots so the AI can learn what to look for")
        else:
            st.markdown("✅ Using red circles from your uploaded image as training examples")
        
        # Show image with training spots
        if pre_marked is None:
            # For clean uploads, show preview with added markers
            preview = image_rgb.copy()
            for spot in st.session_state.training_spots:
                x, y, r = spot
                cv2.circle(preview, (x, y), r, (255, 0, 0), 2)
                cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
        else:
            # For pre-marked uploads, just show the original image (already has red circles)
            preview = image_rgb.copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(preview, width="stretch", caption=f"Training examples: {len(st.session_state.training_spots)}")
        
        with col2:
            st.markdown("### Add Training Example")
            x_input = st.number_input("X coordinate", 0, image.shape[1], 0, key="x_train")
            y_input = st.number_input("Y coordinate", 0, image.shape[0], 0, key="y_train")
            r_input = st.number_input("Radius", 10, 200, 50, key="r_train")
            
            if st.button("➕ Add Example", type="primary"):
                st.session_state.training_spots.append((x_input, y_input, r_input))
                st.rerun()
            
            if st.session_state.training_spots:
                if st.button("🗑️ Clear All"):
                    st.session_state.training_spots = []
                    st.session_state.positive_patches = []
                    st.session_state.negative_patches = []
                    st.rerun()
        
        # Bulk input
        with st.expander("📋 Or paste multiple coordinates"):
            coords_text = st.text_area(
                "Format: x,y,radius (one per line)",
                placeholder="500,300,50\n800,450,50",
                height=100
            )
            if st.button("Add All Examples"):
                if coords_text.strip():
                    count = 0
                    for line in coords_text.strip().split('\n'):
                        try:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) == 3:
                                x, y, r = map(int, parts)
                                st.session_state.training_spots.append((x, y, r))
                                count += 1
                        except:
                            pass
                    if count > 0:
                        st.success(f"✅ Added {count} examples!")
                        st.rerun()
        
        # Only show upload option if user didn't already upload a pre-marked image
        if pre_marked is None:
            with st.expander("📤 Or upload an image with red circles"):
                marked_file = st.file_uploader("Upload marked image", type=["jpg", "jpeg", "png"], key="marked_train")
                
                if marked_file is not None:
                    marked_bytes = np.asarray(bytearray(marked_file.read()), dtype=np.uint8)
                    marked_img = cv2.imdecode(marked_bytes, cv2.IMREAD_COLOR)
                    
                    # Detect red areas
                    hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 50, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 50, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    red_mask = cv2.bitwise_or(mask1, mask2)
                    
                    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if st.button("🔄 Import as Examples"):
                        st.session_state.training_spots = []
                        count = 0
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 20:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    r = max(int(np.sqrt(area / np.pi)) + 5, 50)
                                    st.session_state.training_spots.append((cx, cy, r))
                                    count += 1
                        
                        if count > 0:
                            st.success(f"✅ Imported {count} training examples!")
                            st.rerun()
        else:
            st.info("💡 Red circles were auto-detected from your uploaded image. You can add more examples above if needed.")
        
        # Training section
        if len(st.session_state.training_spots) >= 2:
            st.markdown("---")
            st.header("Step 2: Train AI and Detect Similar Spots")
            
            if st.button("🎓 Train AI on These Examples", type="primary", key="train_btn"):
                with st.spinner("Training AI to recognize dust spots..."):
                    # Create training mask
                    train_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    for x, y, r in st.session_state.training_spots:
                        cv2.circle(train_mask, (x, y), r, 255, -1)
                    
                    # Extract patches
                    pos_patches, neg_patches = create_training_patches(image, train_mask)
                    st.session_state.positive_patches = pos_patches
                    st.session_state.negative_patches = neg_patches
                    
                    st.success(f"✅ Trained on {len(pos_patches)} dust examples and {len(neg_patches)} clean examples!")
            
            if st.session_state.positive_patches:
                if st.button("🔍 Detect Similar Dust Spots", type="primary"):
                    with st.spinner("AI is finding similar dust spots..."):
                        # Run ML detection
                        detected_mask = detect_dust_ml_simple(
                            image,
                            st.session_state.positive_patches,
                            st.session_state.negative_patches,
                            sensitivity=ml_sensitivity
                        )
                        
                        num_spots = cv2.countNonZero(detected_mask)
                        
                        if num_spots > 0:
                            # Show detected spots
                            vis_image = image_rgb.copy()
                            contours, _ = cv2.findContours(detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for contour in contours:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    radius = int(np.sqrt(cv2.contourArea(contour) / np.pi)) + 5
                                    cv2.circle(vis_image, (cx, cy), radius, (0, 255, 0), 2)
                            
                            st.subheader(f"AI Found {len(contours)} Similar Spots!")
                            st.image(vis_image, width="stretch", caption="Green circles = AI detected spots")
                            
                            with st.expander("🔍 View Detection Mask"):
                                st.image(detected_mask, width="stretch")
                            
                            if st.button("✅ Remove All Detected Spots", type="primary"):
                                cleaned_image = remove_dust_spots(image, detected_mask)
                                cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                                st.session_state.cleaned_image = cleaned_rgb
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Original**")
                                    st.image(image_rgb, width="stretch")
                                with col2:
                                    st.markdown("**Cleaned by AI**")
                                    st.image(cleaned_rgb, width="stretch")
                                
                                st.success("✅ AI removed all detected dust spots!")
                                
                                pil_image = Image.fromarray(cleaned_rgb)
                                buf = io.BytesIO()
                                pil_image.save(buf, format="PNG")
                                
                                st.download_button(
                                    label="⬇️ Download Cleaned Image",
                                    data=buf.getvalue(),
                                    file_name="ai_cleaned.png",
                                    mime="image/png"
                                )
                        else:
                            st.warning("No similar spots found. Try lowering sensitivity or adding more varied examples.")
        else:
            st.info("👆 Mark at least 2 dust spots to train the AI")
    
    else:
        # MANUAL MODE
        st.header("Manual Dust Spot Removal")
        
        if 'manual_spots' not in st.session_state:
            st.session_state.manual_spots = []
        
        preview = image_rgb.copy()
        for spot in st.session_state.manual_spots:
            x, y, r = spot
            cv2.circle(preview, (x, y), r, (255, 0, 0), 2)
            cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
        
        st.image(preview, width="stretch", caption=f"Marked: {len(st.session_state.manual_spots)}")
        
        # Manual marking interface (same as before)
        tab1, tab2, tab3 = st.tabs(["📍 Coordinates", "📤 Upload Marked", "📝 Bulk"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                x_m = st.number_input("X", 0, image.shape[1], 0)
            with col2:
                y_m = st.number_input("Y", 0, image.shape[0], 0)
            with col3:
                r_m = st.number_input("R", 5, 200, brush_size)
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕"):
                    st.session_state.manual_spots.append((x_m, y_m, r_m))
                    st.rerun()
        
        with tab2:
            marked_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], key="m_upload")
            if marked_file:
                marked_bytes = np.asarray(bytearray(marked_file.read()), dtype=np.uint8)
                marked_img = cv2.imdecode(marked_bytes, cv2.IMREAD_COLOR)
                hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)
                
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if st.button("Import"):
                    st.session_state.manual_spots = []
                    for contour in contours:
                        if cv2.contourArea(contour) > 20:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                r = max(int(np.sqrt(cv2.contourArea(contour) / np.pi)) + 5, brush_size)
                                st.session_state.manual_spots.append((cx, cy, r))
                    st.rerun()
        
        with tab3:
            coords = st.text_area("x,y,r", height=100)
            if st.button("Add All"):
                for line in coords.strip().split('\n'):
                    try:
                        x, y, r = map(int, [p.strip() for p in line.split(',')])
                        st.session_state.manual_spots.append((x, y, r))
                    except:
                        pass
                st.rerun()
        
        if st.session_state.manual_spots:
            if st.button("🔧 Remove Marked Spots", type="primary"):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for x, y, r in st.session_state.manual_spots:
                    cv2.circle(mask, (x, y), r, 255, -1)
                
                cleaned = remove_dust_spots(image, mask)
                cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_rgb, width="stretch")
                with col2:
                    st.image(cleaned_rgb, width="stretch")
                
                pil_image = Image.fromarray(cleaned_rgb)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                
                st.download_button(
                    "⬇️ Download",
                    data=buf.getvalue(),
                    file_name="cleaned.png",
                    mime="image/png"
                )

else:
    st.info("👆 Upload an image to get started")
    
    st.markdown("---")
    st.subheader("🤖 How Machine Learning Mode Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📝 Mark Examples & Train
        
        **Step 1:** Mark 3-5 dust spots as examples
        
        **Step 2:** Click "Train AI"
        - AI learns color patterns
        - AI learns brightness levels  
        - AI learns texture characteristics
        
        **Step 3:** AI finds similar spots
        - Scans entire image
        - Detects spots matching learned patterns
        - Shows all detected spots
        
        **Step 4:** Review and remove
        - Check if AI found the right spots
        - Adjust sensitivity if needed
        - Remove all detected spots at once!
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Why This Works Better
        
        **Traditional CV:**
        - Fixed algorithms
        - Can't adapt to YOUR dust
        - Misses subtle variations
        
        **Machine Learning:**
        - Learns from YOUR examples
        - Adapts to YOUR specific dust type
        - Finds similar patterns automatically
        - Gets better with more examples
        
        **Perfect for:**
        - Very pale dust spots
        - Unique dust patterns
        - When you have many similar spots
        - When automatic detection fails
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>ML-Powered Dust Removal | Learns from Your Examples</div>", unsafe_allow_html=True)
