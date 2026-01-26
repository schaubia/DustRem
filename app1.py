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
    # This catches out-of-focus dust
    blur1 = cv2.GaussianBlur(gray, (15, 15), 0)
    blur2 = cv2.GaussianBlur(gray, (31, 31), 0)
    dog = cv2.absdiff(blur1, blur2)
    
    # Enhance the difference
    dog_enhanced = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold to find spots
    _, dog_mask = cv2.threshold(dog_enhanced, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Method 2: Detect darker spots in uniform areas
    # Apply median filter to get local background
    median = cv2.medianBlur(gray, 21)
    
    # Find where image is darker than median (dust spots are usually slightly darker)
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
                
                # Accept spots that are somewhat circular (relaxed constraint)
                if circularity > 0.3:  # Reduced from typical 0.7 for more flexibility
                    cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # Dilate slightly to ensure we cover the soft edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    return final_mask

def create_manual_mask_from_clicks(image_shape, click_points, brush_size=30):
    """
    Create a mask from user click points.
    
    Parameters:
    - image_shape: Shape of the image (height, width)
    - click_points: List of (x, y) coordinates
    - brush_size: Size of the circular brush for each click
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for point in click_points:
        cv2.circle(mask, point, brush_size, 255, -1)
    
    return mask

def remove_dust_spots(image, mask):
    """
    Remove dust spots from image using inpainting.
    
    Parameters:
    - image: Input image (BGR format)
    - mask: Binary mask where white pixels indicate dust spots
    """
    # Use Navier-Stokes based inpainting for better results with blurry spots
    result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
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

# Streamlit UI
st.set_page_config(page_title="Sensor Dust Remover", layout="wide")

st.title("📸 Sensor Dust Spot Remover")
st.markdown("""
This application removes **out-of-focus dust spots** from camera sensors that appear as pale, blurry circles 
on uniform backgrounds (like blue skies or gray areas).
""")

# Sidebar for parameters
st.sidebar.header("Detection Settings")

detection_mode = st.sidebar.radio(
    "Detection Mode",
    ["Automatic", "Manual Selection"],
    help="Automatic: Let the algorithm find spots | Manual: Click on spots to mark them"
)

if detection_mode == "Automatic":
    st.sidebar.markdown("### Automatic Detection Parameters")
    
    sensitivity = st.sidebar.slider(
        "Sensitivity",
        min_value=5,
        max_value=50,
        value=15,
        help="Lower = detect fainter spots | Higher = only obvious spots"
    )
    
    min_size = st.sidebar.slider(
        "Minimum Spot Size (pixels)",
        min_value=10,
        max_value=500,
        value=100,
        help="Minimum area of dust spots to detect"
    )
    
    max_size = st.sidebar.slider(
        "Maximum Spot Size (pixels)",
        min_value=500,
        max_value=10000,
        value=5000,
        help="Maximum area of dust spots to detect"
    )
    
    enable_blur_detection = st.sidebar.checkbox(
        "Enhanced blur detection",
        value=True,
        help="Use multiple methods to detect out-of-focus spots"
    )

else:  # Manual mode
    st.sidebar.markdown("### Manual Selection Parameters")
    st.sidebar.info("After uploading, the app will show numbered spots. Enter the numbers of spots you want to remove.")

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
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)
    
    # Process button
    if st.button("🔧 Detect & Remove Dust Spots", type="primary"):
        with st.spinner("Processing..."):
            if detection_mode == "Automatic":
                # Automatic detection
                dust_mask = detect_dust_spots(
                    image,
                    sensitivity=sensitivity,
                    min_area=min_size,
                    max_area=max_size,
                    blur_detection=enable_blur_detection
                )
                
                # Count detected spots
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
                    
                    # Ask for confirmation
                    if st.button("✅ Looks good! Remove these spots"):
                        cleaned_image = remove_dust_spots(image, dust_mask)
                        cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.subheader("Cleaned Image")
                            st.image(cleaned_rgb, use_container_width=True)
                        
                        st.success(f"✅ Successfully removed dust spots!")
                        
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
                    
                    st.warning("⚠️ Not detecting the right spots? Try adjusting the sensitivity and size parameters, or switch to Manual Selection mode.")
                    
                else:
                    st.info("❌ No dust spots detected. Try lowering the sensitivity or reducing the minimum size.")
            
            else:  # Manual mode
                st.info("🖱️ Manual selection mode is coming soon! For now, please use Automatic mode with adjusted parameters.")
                st.markdown("""
                **Tips for better automatic detection:**
                1. Lower the sensitivity to 5-10 for faint spots
                2. Adjust min/max size based on your dust spots
                3. Enable "Enhanced blur detection" for out-of-focus spots
                4. Try on a high-contrast area of your image (blue sky works best)
                """)
    
    # Tips section
    with st.expander("💡 Tips for Detecting Sensor Dust"):
        st.markdown("""
        ### What are sensor dust spots?
        - **Pale, blurry circles** that appear on uniform backgrounds
        - Most visible on **blue skies** or **gray surfaces**
        - **Out of focus** (because dust is on the sensor, not in the scene)
        - Appear in the **same location** across multiple photos
        
        ### Best settings for typical sensor dust:
        1. **Sensitivity**: 10-20 (lower for fainter spots)
        2. **Minimum Size**: 100-200 pixels (for bloated, out-of-focus spots)
        3. **Maximum Size**: 5000-8000 pixels (for large blurry areas)
        4. **Enable blur detection**: ON
        
        ### How to test your camera for sensor dust:
        1. Set your camera to f/16 or smaller aperture
        2. Take a photo of a clear blue sky or white wall
        3. Upload that photo here
        4. Dust spots will be very obvious!
        
        ### If spots aren't detected:
        - Try **lowering sensitivity** to 5-10
        - **Increase minimum size** to 100-200 for bloated spots
        - **Increase maximum size** to 5000-10000 for large areas
        - Make sure you're testing on a **uniform background area**
        """)

else:
    st.info("👆 Please upload an image to get started")
    
    # Example/Demo section
    st.markdown("---")
    st.subheader("About Sensor Dust")
    st.markdown("""
    Sensor dust appears as **pale, blurry spots** on your photos, especially visible on:
    - 🌤️ Blue skies
    - ⬜ Gray or neutral surfaces  
    - 📄 Uniform backgrounds
    
    This tool specifically targets **out-of-focus dust artifacts** that come from:
    - Dust on your camera sensor
    - Dust on the lens (sometimes)
    - Small particles between lens elements
    
    **How it works:**
    1. Upload your image
    2. The algorithm detects blurry, pale spots on uniform areas
    3. Review detected spots
    4. Confirm and download your cleaned image
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with Streamlit | Advanced dust detection for out-of-focus sensor spots
</div>
""", unsafe_allow_html=True)
