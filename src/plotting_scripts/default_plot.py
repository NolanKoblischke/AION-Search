import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI windows
import matplotlib.pyplot as plt
from pathlib import Path
import os

class RescaleToLegacySurvey:
    """Formatter that rescales the images to have a fixed number of bands."""

    def __init__(self):
        pass

    def convert_zeropoint(self, zp: float) -> float:
        return 10.0 ** ((zp - 22.5) / 2.5)

    def reverse_zeropoint(self, scale: float) -> float:
        return 22.5 - 2.5 * np.log10(scale)

    def forward(self, image, survey):
        zpscale = self.convert_zeropoint(27.0) if survey == "HSC" else 1.0
        image = image / zpscale
        return image

    def backward(self, image, survey):
        zpscale = self.reverse_zeropoint(27.0) if survey == "HSC" else 1.0
        image = image * zpscale
        return image

def process_galaxy_image(row_image_array):
    """
    Process galaxy image directly from row['image_array'] to RGB image.
    
    Parameters
    ----------
    row_image_array : np.ndarray
        Raw image array from HDF5 file (row['image_array'])
        
    Returns
    -------
    np.ndarray
        RGB image with shape (npix, npix, 3) ready for display
    """
    # HSC: (5, 144, 144) - g,r,i,z,y bands
    # Legacy: (4, 144, 144) - g,r,i,z bands
    # Legacy North: (3, 152, 152) - g,r,z bands
    
    # Detect survey type and apply HSC rescaling if needed
    rescaler = RescaleToLegacySurvey()
    if row_image_array.shape[0] == 5:  # HSC
        survey = "HSC"
        row_image_array = rescaler.forward(row_image_array, survey)
    else:  # Legacy
        survey = "Legacy"

    # Prepare image for RGB conversion
    if row_image_array.shape[0] == 3:
        img_grz = row_image_array[[0, 1, 2], :, :]  # g, r, z bands from Legacy North
    else:
        img_grz = row_image_array[[0, 1, 3], :, :]  # g, r, z bands from Legacy South and HSC
    
    # Convert to RGB using DECaLS method
    RGB_SCALES = {
        "u": (2, 1.5),
        "g": (2, 6.0),
        "r": (1, 3.4),
        "i": (0, 1.0),
        "z": (0, 2.2),
    }
    
    bands = ("g", "r", "z")
    m = 0.03
    Q = 20.0
    
    # Add batch dimension for processing
    image = np.expand_dims(img_grz, axis=0)
    
    # Collect channel indices and scale factors for the requested bands
    axes, raw_scales = zip(*[RGB_SCALES[b] for b in bands])
    scales = np.array([raw_scales[i] for i in axes], dtype=np.float32)

    # Move channel axis to the end, then reverse it to match `axes`
    img = np.moveaxis(image, 1, -1)        # (batch, npix, npix, 3)
    img = np.flip(img, axis=-1)            # channels now ordered 2,1,0

    # Lupton stretch
    I = np.sum(np.clip(img * scales + m, 0, None), axis=-1) / len(bands)
    I[I == 0] += 1e-6                      # avoid division by zero
    fI = np.arcsinh(Q * I) / np.sqrt(Q)

    img = (img * scales + m) * (fI / I)[..., None]
    img = np.clip(img, 0, 1)

    # Restore original channel position
    rgb_batch = np.moveaxis(img, -1, 1)
    
    # Remove batch dimension and transpose to standard image format
    img_rgb = np.transpose(rgb_batch[0], (1, 2, 0))
    
    return img_rgb

def plot_decals(image_array, object_id, output_dir="plots", script_name="default_plot"):
    """
    Process galaxy image and save it as a PNG file.
    
    Parameters
    ----------
    image_array : np.ndarray
        Raw image array from HDF5 file (row['image_array'])
    object_id : str
        Object ID for naming the output file
    output_dir : str
        Directory to save the PNG file
        
    Returns
    -------
    str
        Path to the saved PNG file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process image to RGB
    img_rgb = process_galaxy_image(image_array)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(img_rgb, origin="upper")
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save PNG file
    output_path = Path(output_dir) / f"{object_id}_{script_name}.png"
    plt.savefig(output_path, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return str(output_path)