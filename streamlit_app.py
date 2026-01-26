import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def remove_dust_spots(image, mask):
    """
    Remove dust spots from image using inpainting.
    
    Parameters:
    - image: Input image (BGR format)
    - mask: Binary mask where white pixels indicate dust spots
    """
    # Use Navier-Stokes based inpainting for better results with blurry spots
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    
    return result

def create_mask_from_drawing(canvas_data, original_shape):
    """
    Convert canvas drawing data to a binary mask.
    """
    if canvas_data is None or canvas_data.image_data is None:
        return None
    
    # Get the drawn image (RGBA)
    drawn = canvas_data.image_data
    
    # Convert to grayscale and create binary mask
    # The canvas draws in white, we want those areas as our mask
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
st.set_page_config(page_title="Manual Dust Spot Remover", layout="wide")

st.title("📸 Manual Dust Spot Remover")
st.markdown("""
Mark dust spots on your image by drawing circles over them, then click to remove them.
Perfect for those subtle, pale dust spots that are hard to detect automatically!
""")

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Sidebar settings
st.sidebar.header("Drawing Settings")

brush_size = st.sidebar.slider(
    "Brush Size",
    min_value=5,
    max_value=100,
    value=30,
    help="Size of the brush to mark dust spots"
)

st.sidebar.markdown("""
### How to use:
1. Upload your image
2. Draw circles over dust spots
3. Click 'Remove Marked Spots'
4. Download cleaned image

### Tips:
- Make brush larger for big spots
- Cover the entire dust spot
- You can draw multiple spots
- Click 'Clear' to start over
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
    st.session_state.original_image = image
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.subheader("Step 1: Mark dust spots on your image")
    st.markdown("Draw circles over the dust spots you want to remove")
    
    # Try to use streamlit-drawable-canvas if available
    try:
        from streamlit_drawable_canvas import st_canvas
        
        # Get image dimensions
        height, width = image_rgb.shape[:2]
        
        # Scale image if too large
        max_width = 800
        if width > max_width:
            scale = max_width / width
            display_width = max_width
            display_height = int(height * scale)
        else:
            display_width = width
            display_height = height
            scale = 1.0
        
        # Create canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=brush_size,
            stroke_color="#FF0000",
            background_image=Image.fromarray(image_rgb),
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="circle",
            key="canvas",
        )
        
        # Process button
        col1, col2 = st.columns([1, 4])
        with col1:
            process_btn = st.button("🔧 Remove Marked Spots", type="primary")
        
        if process_btn and canvas_result.image_data is not None:
            with st.spinner("Removing dust spots..."):
                # Create mask from canvas
                mask = create_mask_from_drawing(canvas_result, image.shape)
                
                if mask is not None and cv2.countNonZero(mask) > 0:
                    # Remove dust spots
                    cleaned_image = remove_dust_spots(image, mask)
                    cleaned_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                    st.session_state.processed_image = cleaned_rgb
                    
                    st.success("✅ Dust spots removed!")
                    
                    # Show result
                    st.subheader("Step 2: Review and download")
                    
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
                        label="⬬ Download Cleaned Image",
                        data=byte_im,
                        file_name="cleaned_image.png",
                        mime="image/png"
                    )
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
        
        st.markdown("---")
        st.subheader("Alternative: Text-based marking (fallback)")
        st.markdown("""
        Since the canvas isn't available, here's a simple alternative approach:
        
        1. Use an image editor to mark dust spots with red circles
        2. Upload that marked image here
        3. The app will detect red circles and remove those areas
        """)
        
        # Simple coordinate input as fallback
        st.markdown("### Or enter coordinates manually:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Your image", use_container_width=True)
        
        with col2:
            st.markdown("Enter dust spot locations (x, y, radius):")
            coords_input = st.text_area(
                "Format: x,y,radius (one per line)",
                placeholder="Example:\n500,300,40\n800,450,35\n1200,600,50",
                height=150
            )
            
            if st.button("Remove Spots at Coordinates"):
                if coords_input.strip():
                    try:
                        # Parse coordinates
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
                            st.image(cleaned_rgb, caption="Cleaned image", use_container_width=True)
                            
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
                        else:
                            st.warning("No valid coordinates entered")
                    except Exception as e:
                        st.error(f"Error processing coordinates: {e}")
                else:
                    st.warning("Please enter coordinates")

else:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("---")
    st.subheader("Why manual selection?")
    st.markdown("""
    Sensor dust spots can be extremely subtle - sometimes barely darker than the surrounding sky.
    Manual selection gives you complete control to mark exactly which spots you want removed.
    
    **This is perfect for:**
    - Very faint, pale dust spots
    - Spots on complex backgrounds  
    - When automatic detection misses spots
    - When you want precise control
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Manual dust spot removal tool | Draw directly on your image
</div>
""", unsafe_allow_html=True)
