import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def detect_dust_spots_advanced(image, sensitivity=20, min_area=50, max_area=5000, detection_method="combined"):
    """
    Advanced dust spot detection optimized for pale, out-of-focus spots on uniform backgrounds.
    Uses multiple sophisticated techniques.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float for better precision
    gray_float = gray.astype(np.float32) / 255.0
    
    if detection_method == "combined" or detection_method == "laplacian":
        # Method 1: Laplacian of Gaussian (LoG) - Excellent for blob detection
        # This is specifically good for out-of-focus circular spots
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=5)
        laplacian_abs = np.abs(laplacian)
        
        # Normalize
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold - lower sensitivity means detect fainter spots
        threshold_value = max(10, 50 - sensitivity)
        _, log_mask = cv2.threshold(laplacian_norm, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        log_mask = np.zeros_like(gray)
    
    if detection_method == "combined" or detection_method == "morphological":
        # Method 2: Top-hat transform - Excellent for detecting small bright or dark features
        # on varying background
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Black top-hat for dark spots
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold
        threshold_value = max(5, 40 - sensitivity)
        _, morph_mask = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        morph_mask = np.zeros_like(gray)
    
    if detection_method == "combined" or detection_method == "variance":
        # Method 3: Local variance filter - Detects areas with low texture (dust spots are uniform)
        # Calculate local standard deviation
        kernel_size = 15
        mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray_float**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # Dust spots have LOW variance (they're uniform blobs)
        # But they differ from their neighborhood
        # So we look for low local variance
        std_dev_uint8 = (std_dev * 255).astype(np.uint8)
        
        # Invert so low variance areas are bright
        inverted_std = 255 - std_dev_uint8
        
        threshold_value = max(150, 255 - sensitivity * 2)
        _, variance_mask = cv2.threshold(inverted_std, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        variance_mask = np.zeros_like(gray)
    
    if detection_method == "combined" or detection_method == "hough":
        # Method 4: Circular Hough Transform - Specifically detects circles
        # Perfect for circular dust spots
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=max(10, 50 - sensitivity),
            minRadius=int(np.sqrt(min_area / np.pi)),
            maxRadius=int(np.sqrt(max_area / np.pi))
        )
        
        hough_mask = np.zeros_like(gray)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(hough_mask, center, radius, 255, -1)
    else:
        hough_mask = np.zeros_like(gray)
    
    # Combine all methods
    if detection_method == "combined":
        combined = cv2.bitwise_or(log_mask, morph_mask)
        combined = cv2.bitwise_or(combined, variance_mask)
        combined = cv2.bitwise_or(combined, hough_mask)
    elif detection_method == "laplacian":
        combined = log_mask
    elif detection_method == "morphological":
        combined = morph_mask
    elif detection_method == "variance":
        combined = variance_mask
    elif detection_method == "hough":
        combined = hough_mask
    else:
        combined = log_mask
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours and filter by size and shape
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Dust spots are reasonably circular
                if circularity > 0.25:  # Very relaxed for irregular dust
                    cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # Dilate to ensure we cover soft edges of out-of-focus dust
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)
    
    return final_mask

def remove_dust_spots(image, mask):
    """Remove dust spots from image using inpainting."""
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    return result

def create_visualization_with_circles(image, mask):
    """Create a visualization showing detected spots with red circles."""
    vis_image = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            radius = int(np.sqrt(cv2.contourArea(contour) / np.pi)) + 5
            cv2.circle(vis_image, (cX, cY), radius, (0, 0, 255), 2)
            cv2.circle(vis_image, (cX, cY), 2, (0, 0, 255), -1)
    
    return vis_image

# Streamlit UI
st.set_page_config(page_title="Dust Spot Remover", layout="wide")

st.title("📸 Advanced Dust Spot Remover")
st.markdown("Remove dust spots using advanced computer vision algorithms or manual marking.")

# Initialize session state
if 'manual_spots' not in st.session_state:
    st.session_state.manual_spots = []
if 'cleaned_image' not in st.session_state:
    st.session_state.cleaned_image = None
if 'show_cleaned' not in st.session_state:
    st.session_state.show_cleaned = False

# Sidebar
st.sidebar.header("Detection Mode")
detection_mode = st.sidebar.radio(
    "Choose mode",
    ["🎯 Automatic Detection (Advanced CV)", "✏️ Manual Selection"],
    help="Automatic: Advanced algorithms find spots | Manual: Mark spots yourself"
)

# Mode-specific parameters
if detection_mode == "🎯 Automatic Detection (Advanced CV)":
    st.sidebar.markdown("### Detection Algorithm")
    
    algorithm = st.sidebar.selectbox(
        "Detection Method",
        ["Combined (Best)", "Laplacian of Gaussian", "Morphological", "Local Variance", "Circular Hough"],
        help="Different algorithms for different types of dust"
    )
    
    algo_map = {
        "Combined (Best)": "combined",
        "Laplacian of Gaussian": "laplacian",
        "Morphological": "morphological",
        "Local Variance": "variance",
        "Circular Hough": "hough"
    }
    
    st.sidebar.markdown("### Detection Parameters")
    sensitivity = st.sidebar.slider("Sensitivity", 1, 50, 25, help="Lower = detect fainter spots")
    min_size = st.sidebar.slider("Minimum Spot Size (pixels)", 10, 500, 100)
    max_size = st.sidebar.slider("Maximum Spot Size (pixels)", 500, 10000, 5000)
    
    st.sidebar.markdown("""
    ### Algorithm Guide:
    **Combined**: Uses all methods together (recommended)
    
    **Laplacian of Gaussian**: Best for blob-like spots
    
    **Morphological**: Best for dark spots on uniform backgrounds
    
    **Local Variance**: Detects uniform areas that differ from surroundings
    
    **Circular Hough**: Specifically finds circular shapes
    """)

else:
    st.sidebar.markdown("### Manual Selection Settings")
    brush_size = st.sidebar.slider("Spot Radius", 10, 150, 40)
    st.sidebar.markdown("""
    ### Three Ways to Mark:
    
    **1. Enter Coordinates**
    - Enter X, Y values
    - Click "Add Spot"
    
    **2. Upload Marked Image**
    - Draw red circles in any editor
    - Upload and auto-detect
    
    **3. Bulk Paste**
    - Paste multiple coordinates
    - Format: x,y,radius
    """)

# File uploader
uploaded_file = st.file_uploader("Upload your image with dust spots", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if detection_mode == "🎯 Automatic Detection (Advanced CV)":
        # AUTOMATIC MODE
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        if st.button("🔧 Detect & Remove Dust Spots", type="primary"):
            with st.spinner(f"Detecting dust spots using {algorithm}..."):
                dust_mask = detect_dust_spots_advanced(
                    image,
                    sensitivity=sensitivity,
                    min_area=min_size,
                    max_area=max_size,
                    detection_method=algo_map[algorithm]
                )
                
                num_spots = cv2.countNonZero(dust_mask)
                
                if num_spots > 0:
                    vis_image = create_visualization_with_circles(image, dust_mask)
                    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    
                    st.subheader(f"Detected Spots: {len(cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])} spots found")
                    st.image(vis_rgb, use_container_width=True)
                    
                    with st.expander("🔍 View Detection Mask"):
                        st.image(dust_mask, caption="White areas will be removed", use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Remove These Spots", type="primary"):
                            with st.spinner("Removing dust spots..."):
                                cleaned_image = remove_dust_spots(image, dust_mask)
                                cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                                st.session_state.cleaned_image = cleaned_rgb
                                st.session_state.show_cleaned = True
                                st.rerun()
                    
                    with col_b:
                        if st.button("❌ Not Good - Try Different Algorithm"):
                            st.info("Try a different algorithm from the sidebar or adjust sensitivity")
                    
                    # Show cleaned if available
                    if st.session_state.show_cleaned and st.session_state.cleaned_image is not None:
                        with col2:
                            st.subheader("Cleaned Image")
                            st.image(st.session_state.cleaned_image, use_container_width=True)
                        
                        st.success("✅ Dust spots removed successfully!")
                        
                        pil_image = Image.fromarray(st.session_state.cleaned_image)
                        buf = io.BytesIO()
                        pil_image.save(buf, format="PNG")
                        
                        st.download_button(
                            label="⬇️ Download Cleaned Image",
                            data=buf.getvalue(),
                            file_name="cleaned_image.png",
                            mime="image/png",
                            key="auto_download"
                        )
                else:
                    st.warning(f"❌ No dust spots detected with {algorithm}")
                    st.info("""
                    **Try these adjustments:**
                    - Lower the sensitivity (try 5-15)
                    - Reduce minimum size to 50
                    - Try a different detection algorithm
                    - Or switch to Manual Selection mode
                    """)
    
    else:
        # MANUAL MODE
        st.subheader("Manual Dust Spot Removal")
        
        # Create preview with marked spots
        preview = image_rgb.copy()
        for spot in st.session_state.manual_spots:
            x, y, r = spot
            cv2.circle(preview, (x, y), r, (255, 0, 0), 2)
            cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
        
        st.image(preview, use_container_width=True, caption=f"Marked spots: {len(st.session_state.manual_spots)}")
        
        # Three methods
        tab1, tab2, tab3 = st.tabs(["📍 Enter Coordinates", "📤 Upload Marked Image", "📝 Bulk Paste"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                x_input = st.number_input("X coordinate", 0, image.shape[1], 0, key="x_manual")
            with col2:
                y_input = st.number_input("Y coordinate", 0, image.shape[0], 0, key="y_manual")
            with col3:
                r_input = st.number_input("Radius", 5, 200, brush_size, key="r_manual")
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Add", type="primary"):
                    st.session_state.manual_spots.append((x_input, y_input, r_input))
                    st.rerun()
        
        with tab2:
            st.markdown("**Step 1:** Download your image to mark it")
            pil_original = Image.fromarray(image_rgb)
            buf = io.BytesIO()
            pil_original.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Original Image",
                data=buf.getvalue(),
                file_name="original.png",
                mime="image/png"
            )
            
            st.markdown("**Step 2:** Draw RED circles in any image editor")
            st.markdown("**Step 3:** Upload the marked image")
            
            marked_file = st.file_uploader("Upload marked image", type=["jpg", "jpeg", "png"], key="marked")
            
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
                
                st.image(red_mask, caption="Detected red areas", use_container_width=True)
                
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if st.button("🔄 Import Detected Spots", type="primary"):
                    st.session_state.manual_spots = []
                    count = 0
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 20:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                r = max(int(np.sqrt(area / np.pi)) + 5, brush_size)
                                st.session_state.manual_spots.append((cx, cy, r))
                                count += 1
                    
                    if count > 0:
                        st.success(f"✅ Imported {count} spots!")
                        st.rerun()
                    else:
                        st.warning("No red spots detected. Use bright red (255,0,0)")
        
        with tab3:
            coords_text = st.text_area(
                "Paste coordinates (format: x,y,radius - one per line)",
                placeholder="500,300,40\n800,450,35\n1200,600,50",
                height=150
            )
            if st.button("Add All Coordinates", type="primary"):
                if coords_text.strip():
                    count = 0
                    for line in coords_text.strip().split('\n'):
                        try:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) == 3:
                                x, y, r = map(int, parts)
                                st.session_state.manual_spots.append((x, y, r))
                                count += 1
                        except:
                            pass
                    if count > 0:
                        st.success(f"✅ Added {count} spots!")
                        st.rerun()
        
        # Show controls
        if st.session_state.manual_spots:
            st.markdown(f"**📍 Total marked spots: {len(st.session_state.manual_spots)}**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔧 Remove All Marked Spots", type="primary"):
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    for x, y, r in st.session_state.manual_spots:
                        cv2.circle(mask, (x, y), r, 255, -1)
                    
                    cleaned_image = remove_dust_spots(image, mask)
                    cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                    st.session_state.cleaned_image = cleaned_rgb
                    st.session_state.show_cleaned = True
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Marks"):
                    st.session_state.manual_spots = []
                    st.session_state.show_cleaned = False
                    st.rerun()
            
            # Show cleaned if available
            if st.session_state.show_cleaned and st.session_state.cleaned_image is not None:
                st.success("✅ Dust spots removed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.image(image_rgb, use_container_width=True)
                with col2:
                    st.markdown("**Cleaned**")
                    st.image(st.session_state.cleaned_image, use_container_width=True)
                
                pil_image = Image.fromarray(st.session_state.cleaned_image)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                
                st.download_button(
                    label="⬇️ Download Cleaned Image",
                    data=buf.getvalue(),
                    file_name="cleaned_image.png",
                    mime="image/png",
                    key="manual_download"
                )

else:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("---")
    st.subheader("Advanced Detection Algorithms")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Automatic Detection
        **5 Advanced Algorithms:**
        - **Laplacian of Gaussian**: Blob detection
        - **Morphological Top-Hat**: Dark spot detection
        - **Local Variance**: Uniform region detection
        - **Circular Hough Transform**: Circle detection
        - **Combined**: All methods together
        """)
    
    with col2:
        st.markdown("""
        ### ✏️ Manual Selection
        **Perfect for:**
        - Very subtle dust spots
        - When automatic fails
        - Complete control
        - Precise removal
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Advanced Dust Spot Remover with Multiple CV Algorithms</div>", unsafe_allow_html=True)
