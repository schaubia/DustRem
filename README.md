# Dust Artifact Remover - Streamlit App

A Streamlit application that automatically detects and removes dust spots and artifacts from photos caused by dust on camera sensors or lenses, while preserving image detail.

## Features

- **Automatic Dust Detection**: Uses computer vision to identify dust spots
- **Detail Preservation**: Advanced inpainting algorithms maintain image quality
- **Adjustable Parameters**: Fine-tune detection sensitivity and spot size
- **Visual Feedback**: See detected spots before and after cleaning
- **Easy Download**: Save your cleaned images instantly

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run dust_removal_app.py
   ```

3. **Access the app:**
   - The app will automatically open in your default browser
   - Or navigate to `http://localhost:8501`

## How to Use

1. **Upload Image**: Click "Choose an image..." and select your photo
2. **Adjust Settings** (optional): Use the sidebar to fine-tune detection
   - **Sensitivity**: Controls how dark a spot must be to be detected (10-50)
   - **Minimum Spot Size**: Ignore spots smaller than this (1-20 pixels)
   - **Maximum Spot Size**: Ignore spots larger than this (50-1000 pixels)
3. **Process**: Click "Remove Dust Spots" button
4. **Download**: Save your cleaned image using the download button

## Parameter Guide

### Sensitivity
- **Lower values (10-20)**: Detect even faint dust spots
- **Default (30)**: Good balance for most images
- **Higher values (40-50)**: Only detect very dark, prominent spots

### Spot Size
- Adjust **Minimum Size** if detecting too much noise/texture
- Adjust **Maximum Size** if large artifacts aren't being removed
- Default values work well for typical sensor dust

## Technical Details

The application uses:
- **OpenCV** for image processing
- **Adaptive thresholding** for dust spot detection
- **Morphological operations** for noise reduction
- **Telea inpainting algorithm** for detail-preserving removal

## Tips for Best Results

- Use high-resolution images when possible
- Works best on photos with good contrast
- Try photographing a blank wall or sky to see sensor dust clearly
- Adjust parameters if initial results aren't perfect
- The algorithm preserves details better than simple clone/heal tools

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Troubleshooting

**No spots detected?**
- Lower the sensitivity value
- Reduce the minimum spot size
- Check that your image actually has visible dust spots

**Too many false detections?**
- Increase sensitivity value
- Increase minimum spot size
- The spots might be actual image features, not dust

**Processing is slow?**
- Large images take longer to process
- Consider resizing very large images before upload

## License

This is a demonstration application for educational and personal use.
