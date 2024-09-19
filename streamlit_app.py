import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import streamlit as st
from pydicom.data import get_testdata_file


# Streamlit app title
st.title("DICOM CT Slices and RT Files Uploader with Independent Slice Scrolling")

# Temporary directory for saving uploaded files
temp_dir = "temp_dicom"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Drag and drop file uploader
uploaded_files = st.file_uploader(
    "Upload DICOM CT Slices, RT Dose, RT Structure, and RT Plan files",
    type=["dcm"],
    accept_multiple_files=True
)

# Save uploaded files to the temp folder
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {len(uploaded_files)} files successfully!")

# Process and display CT slices with a slider
def load_and_display_ct_slices(temp_dir):
    # Load the DICOM files
    dicom_files = glob.glob(os.path.join(temp_dir, "*.dcm"))
    files = []
    for fname in dicom_files:
        files.append(pydicom.dcmread(fname))

    if not files:
        st.error("No DICOM files found.")
        return

    # Skip files with no SliceLocation (e.g., scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount += 1

    if not slices:
        st.error("No valid CT slices found.")
        return

    # Ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # Pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ss / ps[0]  # Corrected aspect for sagittal view
    cor_aspect = ss / ps[0]

    # Create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # Fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # Add sliders for axial, sagittal, and coronal slices
    axial_slider = st.slider("Select Axial Slice", 0, img_shape[2] - 1, img_shape[2] // 2)
    sagittal_slider = st.slider("Select Sagittal Slice", 0, img_shape[0] - 1, img_shape[0] // 2)
    coronal_slider = st.slider("Select Coronal Slice", 0, img_shape[1] - 1, img_shape[1] // 2)

    # Plot 3 orthogonal slices with independent sliders
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice
    axes[0].imshow(img3d[:, :, axial_slider], cmap='gray')
    axes[0].set_aspect(ax_aspect)
    axes[0].set_title(f"Axial Slice {axial_slider+1}/{img_shape[2]}")

    # Sagittal slice (now rotated and properly scaled)
    sagittal_view = np.rot90(img3d[sagittal_slider, :, :])
    axes[1].imshow(sagittal_view, cmap='gray')
    axes[1].set_aspect(sag_aspect)
    axes[1].set_title(f"Sagittal Slice {sagittal_slider+1}/{img_shape[0]}")

    # Coronal slice (proper scaling and orientation)
    coronal_view = np.rot90(img3d[:, coronal_slider, :])
    axes[2].imshow(coronal_view, cmap='gray')
    axes[2].set_aspect(cor_aspect)
    axes[2].set_title(f"Coronal Slice {coronal_slider+1}/{img_shape[1]}")

    # Remove axes labels
    for ax in axes:
        ax.axis('off')

    st.pyplot(fig)

# Display CT slices if files are uploaded
if uploaded_files:
    load_and_display_ct_slices(temp_dir)
else:
    st.info("Please upload DICOM CT slices, RT Dose, RT Structure, and RT Plan files.")

# Function to list the RT Plan beams
def list_beams(ds: pydicom.Dataset) -> str:
    """Summarizes the RTPLAN beam information in the dataset."""
    lines = [f"{'Beam name':^13s} {'Number':^8s} {'Gantry':^8s} {'SSD (cm)':^11s}"]
    for beam in ds.BeamSequence:
        cp0 = beam.ControlPointSequence[0]
        ssd = float(cp0.SourceToSurfaceDistance / 10)
        lines.append(
            f"{beam.BeamName:^13s} {beam.BeamNumber:8d} {cp0.GantryAngle:8.1f} {ssd:8.1f}"
        )
    return "\n".join(lines)

# Function to load RT Plan and extract tags
def load_rt_plan_and_extract_tags(temp_dir):
    rt_plan_file = None
    # Find RT Plan file in the temp directory
    for file_name in os.listdir(temp_dir):
        if "RTPLAN" in file_name.upper():
            rt_plan_file = os.path.join(temp_dir, file_name)
            break

    if rt_plan_file:
        # Read the RT Plan DICOM file
        ds = pydicom.dcmread(rt_plan_file)
        # Display the beam information
        beam_info = list_beams(ds)
        st.text("RT Plan Beam Information:")
        st.text(beam_info)
    else:
        st.error("RT Plan file not found.")

# Call the function to read and display RT Plan information if files are uploaded
if uploaded_files:
    st.header("RT Plan Information")
    load_rt_plan_and_extract_tags(temp_dir)
else:
    st.info("Please upload the RT Plan file.")
