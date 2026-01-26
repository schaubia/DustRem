import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def detect_dust_spots(image, sensitivity=20, min_area=50, max_area=2000, blur_detection=True):
    """
    Detect out-of-focus dust spots that appear as pale, blurry circles on uniform backgrounds.
    
    Parameters:
    - image: Input image (BGR format)
    - sensitivity: How different from surrounding area (lower = more sensitive)
    - min_area: Minimum area of dust spots to detect
    - max_area: Maximum area of dust spots to detect
    - blur_detection: If True, specifically looks for blurry/out-of-focus spots
    """
    # Convert to LAB color space for better detection on blue/gray areas
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Also work with grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect low-contrast blurry spots using difference of Gaussians
    blur1 = cv2.GaussianBlur(gray, (15, 15), 0)
    blur2 = cv2.GaussianBlur(gray, (31, 31), 0)
    dog = cv2.absdiff(blur1, blur2)
    
    # Enhance the difference
    dog_enhanced = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold to find spots
    _, dog_mask = cv2.threshold(dog_enhanced, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Method 2: Detect darker spots in uniform areas
    median = cv2.medianBlur(gray, 21)
    diff = cv2.absdiff(gray, median)
    _, dark_mask = cv2.threshold(diff, sensitivity // 2, 255, cv2.THRESH_BINARY)
    
    # Method 3: Detect based on luminance variation in L channel
    l_blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
    l_diff = cv2.absdiff(l_channel, l_blur)
    _, l_mask = cv2.threshold(l_diff, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Combine all methods
    if blur_detection:
        combined = cv2.bitwise_or(dog_mask, dark_mask)
        combined = cv2.bitwise_or(combined, l_mask)
    else:
        combined = dark_mask
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours and filter by shape and size
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask
    final_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            # Check circularity - dust spots are usually round
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Accept spots that are somewhat circular
                if circularity > 0.3:
                    cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # Dilate slightly to ensure we cover the soft edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    return final_mask

def remove_dust_spots(image, mask):
    """
    Remove dust spots from image using inpainting.
    """
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    return result

def create_visualization_with_circles(image, mask):
    """
    Create a visualization showing detected spots with red circles.
    """
    vis_image = image.copy()
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around detected spots
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            radius = int(np.sqrt(cv2.contourArea(contour) / np.pi)) + 5
            cv2.circle(vis_image, (cX, cY), radius, (0, 0, 255), 2)
    
    return vis_image

def create_mask_from_drawing(canvas_data, original_shape):
    """
    Convert canvas drawing data to a binary mask.
    """
    if canvas_data is None or canvas_data.image_data is None:
        return None
    
    # Get the drawn image (RGBA)
    drawn = canvas_data.image_data
    
    # Convert to grayscale and create binary mask
    if len(drawn.shape) == 3:
        gray = cv2.cvtColor(drawn, cv2.COLOR_RGBA2GRAY)
    else:
        gray = drawn
    
    # Any non-zero pixel is part of the mask
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Resize mask to match original image size if needed
    if mask.shape[:2] != original_shape[:2]:
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask

# Streamlit UI
st.set_page_config(page_title="Dust Spot Remover", layout="wide")

st.title("📸 Dust Spot Remover")
st.markdown("""
Remove dust spots from your photos with automatic detection or manual marking.
""")

# Sidebar for mode selection
st.sidebar.header("Detection Mode")

detection_mode = st.sidebar.radio(
    "Choose mode",
    ["🎯 Automatic Detection", "✏️ Manual Selection"],
    help="Automatic: Let AI find spots | Manual: Draw circles over spots yourself"
)

# Mode-specific parameters
if detection_mode == "🎯 Automatic Detection":
    st.sidebar.markdown("### Automatic Detection Settings")
    
    sensitivity = st.sidebar.slider(
        "Sensitivity",
        min_value=5,
        max_value=50,
        value=15,
        help="Lower = detect fainter spots"
    )
    
    min_size = st.sidebar.slider(
        "Minimum Spot Size (pixels)",
        min_value=10,
        max_value=500,
        value=100,
        help="Minimum area of dust spots"
    )
    
    max_size = st.sidebar.slider(
        "Maximum Spot Size (pixels)",
        min_value=500,
        max_value=10000,
        value=5000,
        help="Maximum area of dust spots"
    )
    
    enable_blur_detection = st.sidebar.checkbox(
        "Enhanced blur detection",
        value=True,
        help="Use multiple methods to detect out-of-focus spots"
    )

else:  # Manual mode
    st.sidebar.markdown("### Manual Selection Settings")
    
    brush_size = st.sidebar.slider(
        "Default Spot Radius",
        min_value=10,
        max_value=150,
        value=40,
        help="Default size for dust spot circles"
    )
    
    st.sidebar.markdown("""
    ### How to use manual mode:
    
    **Method 1: Enter coordinates**
    - Find spot coordinates in image viewer
    - Enter x, y, radius
    - Click "Add Spot"
    
    **Method 2: Upload marked image**
    - Download original image
    - Draw RED circles in any editor
    - Upload marked image
    - Spots auto-detected!
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Upload your image with dust spots",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if detection_mode == "🎯 Automatic Detection":
        # AUTOMATIC MODE
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        if st.button("🔧 Detect & Remove Dust Spots", type="primary"):
            with st.spinner("Detecting dust spots..."):
                dust_mask = detect_dust_spots(
                    image,
                    sensitivity=sensitivity,
                    min_area=min_size,
                    max_area=max_size,
                    blur_detection=enable_blur_detection
                )
                
                num_spots = cv2.countNonZero(dust_mask)
                
                if num_spots > 0:
                    # Create visualization
                    vis_image = create_visualization_with_circles(image, dust_mask)
                    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    
                    st.subheader("Detected Spots (marked in red)")
                    st.image(vis_rgb, use_container_width=True)
                    
                    # Show detection mask
                    with st.expander("🔍 View Detection Mask"):
                        st.image(dust_mask, caption="White areas will be removed", use_container_width=True)
                    
                    # Confirm and remove
                    if st.button("✅ Remove These Spots"):
                        cleaned_image = remove_dust_spots(image, dust_mask)
                        cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.subheader("Cleaned Image")
                            st.image(cleaned_rgb, use_container_width=True)
                        
                        st.success("✅ Dust spots removed!")
                        
                        # Download button
                        pil_image = Image.fromarray(cleaned_rgb)
                        buf = io.BytesIO()
                        pil_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="⬇️ Download Cleaned Image",
                            data=byte_im,
                            file_name="cleaned_image.png",
                            mime="image/png"
                        )
                else:
                    st.info("❌ No dust spots detected. Try lowering sensitivity or switch to Manual Selection mode.")
    
    else:
        # MANUAL MODE - Simple coordinate-based selection
        st.subheader("Manual Dust Spot Removal")
        
        # Initialize session state for manual marks
        if 'manual_spots' not in st.session_state:
            st.session_state.manual_spots = []
        if 'preview_image' not in st.session_state:
            st.session_state.preview_image = None
        
        # Display image with current markings
        if st.session_state.manual_spots:
            # Create preview with circles
            preview = image_rgb.copy()
            for spot in st.session_state.manual_spots:
                x, y, r = spot
                cv2.circle(preview, (x, y), r, (255, 0, 0), 2)
                cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
            st.session_state.preview_image = preview
        else:
            st.session_state.preview_image = image_rgb.copy()
        
        st.image(st.session_state.preview_image, caption="Your image with marked spots", use_container_width=True)
        
        # Input method selection
        input_method = st.radio(
            "How to mark spots:",
            ["Enter coordinates manually", "Upload marked image"],
            horizontal=True
        )
        
        if input_method == "Enter coordinates manually":
            st.markdown("""
            **How to find coordinates:**
            1. Open your image in any image viewer (Windows Photos, Preview on Mac, etc.)
            2. Hover over a dust spot to see coordinates (usually shown at bottom)
            3. Enter the x, y coordinates and radius below
            """)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                x_coord = st.number_input("X coordinate", min_value=0, max_value=image.shape[1], value=0, step=1)
            with col2:
                y_coord = st.number_input("Y coordinate", min_value=0, max_value=image.shape[0], value=0, step=1)
            with col3:
                radius = st.number_input("Radius", min_value=5, max_value=200, value=brush_size, step=5)
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Add Spot"):
                    st.session_state.manual_spots.append((x_coord, y_coord, radius))
                    st.rerun()
            
            # Bulk input option
            with st.expander("📝 Or enter multiple spots at once"):
                coords_input = st.text_area(
                    "Format: x,y,radius (one per line)",
                    placeholder="Example:\n500,300,40\n800,450,35\n1200,600,50",
                    height=150
                )
                
                if st.button("Add All Spots"):
                    if coords_input.strip():
                        try:
                            for line in coords_input.strip().split('\n'):
                                parts = line.strip().split(',')
                                if len(parts) == 3:
                                    x, y, r = map(int, parts)
                                    st.session_state.manual_spots.append((x, y, r))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error parsing coordinates: {e}")
        
        else:  # Upload marked image
            st.markdown("""
            **Upload a marked version of your image:**
            1. Download your original image below
            2. Open it in any image editor (Paint, Photoshop, etc.)
            3. Draw RED circles over dust spots
            4. Upload the marked image here
            """)
            
            # Download original for marking
            pil_original = Image.fromarray(image_rgb)
            buf = io.BytesIO()
            pil_original.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Original to Mark",
                data=buf.getvalue(),
                file_name="original_to_mark.png",
                mime="image/png"
            )
            
            marked_file = st.file_uploader(
                "Upload your marked image (with red circles)",
                type=["jpg", "jpeg", "png"],
                key="marked_upload"
            )
            
            if marked_file is not None:
                # Read marked image
                marked_bytes = np.asarray(bytearray(marked_file.read()), dtype=np.uint8)
                marked_img = cv2.imdecode(marked_bytes, cv2.IMREAD_COLOR)
                
                # Detect red circles in the marked image
                hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)
                
                # Red color range
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Find contours
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detected_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum size
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            r = int(np.sqrt(area / np.pi)) + 10
                            st.session_state.manual_spots.append((cx, cy, r))
                            detected_count += 1
                
                if detected_count > 0:
                    st.success(f"✅ Detected {detected_count} marked spots!")
                    st.rerun()
                else:
                    st.warning("⚠️ No red markings detected. Make sure to use bright red color.")
        
        # Show current spots
        if st.session_state.manual_spots:
            st.markdown(f"**📍 Marked spots: {len(st.session_state.manual_spots)}**")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear All"):
                    st.session_state.manual_spots = []
                    st.session_state.preview_image = None
                    st.rerun()
            
            # Show list of spots with delete option
            with st.expander("View/Edit marked spots"):
                for idx, (x, y, r) in enumerate(st.session_state.manual_spots):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"Spot {idx+1}: x={x}, y={y}, radius={r}")
                    with col2:
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.manual_spots.pop(idx)
                            st.rerun()
        
        # Remove spots button
        if st.session_state.manual_spots:
            if st.button("🔧 Remove Marked Spots", type="primary"):
                with st.spinner("Removing dust spots..."):
                    # Create mask from manual spots
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    
                    for x, y, r in st.session_state.manual_spots:
                        cv2.circle(mask, (x, y), r, 255, -1)
                    
                    # Remove dust spots
                    cleaned_image = remove_dust_spots(image, mask)
                    cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                    
                    st.success("✅ Dust spots removed!")
                    
                    # Show result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original**")
                        st.image(image_rgb, use_container_width=True)
                    with col2:
                        st.markdown("**Cleaned**")
                        st.image(cleaned_rgb, use_container_width=True)
                    
                    # Download button
                    pil_image = Image.fromarray(cleaned_rgb)
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="⬇️ Download Cleaned Image",
                        data=byte_im,
                        file_name="cleaned_image.png",
                        mime="image/png"
                    )
        else:
            st.info("👆 Add some dust spot locations above to get started")

else:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("---")
    st.subheader("Choose Your Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Automatic Detection
        - AI finds dust spots automatically
        - Best for obvious spots
        - Adjust sensitivity settings
        - Quick and easy
        """)
    
    with col2:
        st.markdown("""
        ### ✏️ Manual Selection  
        - You mark spots yourself
        - Perfect for subtle, pale spots
        - Complete control
        - Best for tricky cases
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Dust Spot Remover | Automatic & Manual Modes
</div>
""", unsafe_allow_html=True)
