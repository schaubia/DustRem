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
        "Brush Size",
        min_value=10,
        max_value=150,
        value=40,
        help="Size of the circle to mark dust spots"
    )
    
    st.sidebar.markdown("""
    ### How to use manual mode:
    1. Upload your image
    2. Click to place circles over dust spots
    3. Adjust circle size if needed
    4. Click 'Remove Marked Spots'
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
        # MANUAL MODE
        st.subheader("Mark dust spots on your image")
        st.markdown("Click to place circles over dust spots")
        
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Get image dimensions
            height, width = image_rgb.shape[:2]
            
            # Scale image if too large for display
            max_canvas_width = 800
            if width > max_canvas_width:
                scale = max_canvas_width / width
                canvas_width = max_canvas_width
                canvas_height = int(height * scale)
                display_image = cv2.resize(image_rgb, (canvas_width, canvas_height))
            else:
                canvas_width = width
                canvas_height = height
                display_image = image_rgb
            
            # Create canvas with PIL Image
            pil_bg = Image.fromarray(display_image)
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=brush_size,
                stroke_color="#FF0000",
                background_image=pil_bg,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="circle",
                point_display_radius=0,
                key="canvas",
            )
            
            # Process button
            if st.button("🔧 Remove Marked Spots", type="primary"):
                if canvas_result.image_data is not None:
                    with st.spinner("Removing dust spots..."):
                        # Create mask from canvas
                        mask = create_mask_from_drawing(canvas_result, image.shape)
                        
                        if mask is not None and cv2.countNonZero(mask) > 0:
                            # Scale mask back to original size if needed
                            if mask.shape[:2] != image.shape[:2]:
                                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
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
                            st.warning("⚠️ No areas marked. Please draw circles over the dust spots.")
                else:
                    st.warning("⚠️ No areas marked. Please draw circles over the dust spots.")
        
        except ImportError:
            st.error("""
            ❌ **streamlit-drawable-canvas is not installed.**
            
            Please install it by running:
            ```
            pip install streamlit-drawable-canvas
            ```
            
            Then restart your Streamlit app.
            """)
            
            # Fallback: coordinate input
            st.markdown("---")
            st.subheader("Alternative: Enter coordinates manually")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Your image", use_container_width=True)
            
            with col2:
                st.markdown("Enter dust spot locations (x, y, radius):")
                coords_input = st.text_area(
                    "Format: x,y,radius (one per line)",
                    placeholder="Example:\n500,300,40\n800,450,35",
                    height=150
                )
                
                if st.button("Remove Spots at Coordinates"):
                    if coords_input.strip():
                        try:
                            mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            
                            for line in coords_input.strip().split('\n'):
                                parts = line.strip().split(',')
                                if len(parts) == 3:
                                    x, y, radius = map(int, parts)
                                    cv2.circle(mask, (x, y), radius, 255, -1)
                            
                            if cv2.countNonZero(mask) > 0:
                                cleaned_image = remove_dust_spots(image, mask)
                                cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                                
                                st.success("✅ Spots removed!")
                                st.image(cleaned_rgb, use_container_width=True)
                                
                                # Download
                                pil_image = Image.fromarray(cleaned_rgb)
                                buf = io.BytesIO()
                                pil_image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="⬇️ Download",
                                    data=byte_im,
                                    file_name="cleaned_image.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.error(f"Error: {e}")

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
