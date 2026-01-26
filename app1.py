import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def detect_dust_spots(image, threshold=30, min_area=5, max_area=500):
    """
    Detect dust spots in an image using adaptive thresholding and morphological operations.
    
    Parameters:
    - image: Input image (BGR format)
    - threshold: Threshold for detecting dark spots
    - min_area: Minimum area of dust spots to detect
    - max_area: Maximum area of dust spots to detect
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create a mask for dark spots
    # Invert the image so dark spots become bright
    inverted = cv2.bitwise_not(blurred)
    
    # Apply adaptive thresholding to detect local dark areas
    adaptive_thresh = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, -threshold
    )
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of dust spots
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask based on size filtering
    final_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # Dilate the mask slightly to ensure we cover the entire dust spot
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    return final_mask

def remove_dust_spots(image, mask):
    """
    Remove dust spots from image using inpainting.
    
    Parameters:
    - image: Input image (BGR format)
    - mask: Binary mask where white pixels indicate dust spots
    """
    # Use Telea inpainting algorithm for better detail preservation
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

def create_comparison_image(original, processed):
    """Create a side-by-side comparison image."""
    # Ensure both images have the same height
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    if h1 != h2:
        # Resize to match heights
        scale = h1 / h2
        processed = cv2.resize(processed, (int(w2 * scale), h1))
    
    # Concatenate horizontally
    comparison = np.hstack([original, processed])
    return comparison

# Streamlit UI
st.set_page_config(page_title="Dust Artifact Remover", layout="wide")

st.title("📸 Dust Artifact Remover")
st.markdown("""
This application removes dust spots and artifacts from photos while preserving image detail.
Upload a photo with dust spots from your camera sensor or lens, and the app will automatically detect and remove them.
""")

# Sidebar for parameters
st.sidebar.header("Detection Parameters")
st.sidebar.markdown("Adjust these settings to fine-tune dust spot detection:")

sensitivity = st.sidebar.slider(
    "Sensitivity",
    min_value=10,
    max_value=50,
    value=30,
    help="Lower values detect lighter spots, higher values only detect darker spots"
)

min_size = st.sidebar.slider(
    "Minimum Spot Size (pixels)",
    min_value=1,
    max_value=20,
    value=5,
    help="Minimum size of dust spots to detect"
)

max_size = st.sidebar.slider(
    "Maximum Spot Size (pixels)",
    min_value=50,
    max_value=1000,
    value=500,
    help="Maximum size of dust spots to detect"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
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
    if st.button("🔧 Remove Dust Spots", type="primary"):
        with st.spinner("Detecting and removing dust spots..."):
            # Detect dust spots
            dust_mask = detect_dust_spots(
                image,
                threshold=sensitivity,
                min_area=min_size,
                max_area=max_size
            )
            
            # Count detected spots
            num_spots = cv2.countNonZero(dust_mask)
            
            if num_spots > 0:
                # Remove dust spots
                cleaned_image = remove_dust_spots(image, dust_mask)
                cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Cleaned Image")
                    st.image(cleaned_rgb, use_container_width=True)
                
                st.success(f"✅ Successfully processed! Detected and removed dust artifacts.")
                
                # Show the mask
                with st.expander("View Detected Dust Spots"):
                    st.image(dust_mask, caption="Dust Spot Mask (white areas will be removed)", use_container_width=True)
                
                # Download button
                # Convert to PIL Image for saving
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
                st.info("No dust spots detected with current settings. Try adjusting the parameters in the sidebar.")
    
    # Tips section
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - **Sensitivity**: Start with default (30) and adjust if needed
            - Lower values: Detect lighter/fainter dust spots
            - Higher values: Only detect darker, more prominent spots
        
        - **Spot Size**: Adjust based on your image
            - Increase minimum size if detecting too much noise
            - Increase maximum size if large artifacts aren't being removed
        
        - **Image Quality**: Works best with:
            - High-resolution images
            - Good contrast between dust spots and background
            - Evenly lit photos
        
        - **Processing**: The app uses advanced inpainting algorithms to fill in removed spots
          while preserving surrounding detail and texture
        """)

else:
    st.info("👆 Please upload an image to get started")
    
    # Example/Demo section
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    1. **Upload** your image with dust artifacts
    2. **Adjust** detection parameters if needed (optional)
    3. **Click** "Remove Dust Spots" to process
    4. **Download** your cleaned image
    
    The application uses computer vision techniques to:
    - Detect dark spots and artifacts automatically
    - Intelligently fill in the removed areas using surrounding pixels
    - Preserve fine details and image quality
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with Streamlit | Uses OpenCV for image processing
</div>
""", unsafe_allow_html=True)
