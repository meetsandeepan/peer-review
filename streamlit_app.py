import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import streamlit as st
from scipy.ndimage import zoom, rotate

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

# Function to load DICOM CT slices
def load_ct_slices(temp_dir):
    dicom_files = glob.glob(os.path.join(temp_dir, "*.dcm"))
    slices = [pydicom.dcmread(fname) for fname in dicom_files if hasattr(pydicom.dcmread(fname), "SliceLocation")]
    
    if not slices:
        st.error("No valid CT slices found.")
        return None, None

    slices.sort(key=lambda s: s.SliceLocation)
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ss / ps[0]
    cor_aspect = ss / ps[0]

    img_shape = list(slices[0].pixel_array.shape) + [len(slices)]
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img3d[:, :, i] = s.pixel_array

    return img3d, (ax_aspect, sag_aspect, cor_aspect)

# Function to display slices
def display_ct_slices(img3d, aspects):
    axial_slider = st.slider("Select Axial Slice", 0, img3d.shape[2] - 1, img3d.shape[2] // 2)
    sagittal_slider = st.slider("Select Sagittal Slice", 0, img3d.shape[0] - 1, img3d.shape[0] // 2)
    coronal_slider = st.slider("Select Coronal Slice", 0, img3d.shape[1] - 1, img3d.shape[1] // 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img3d[:, :, axial_slider], cmap='gray')
    axes[0].set_aspect(aspects[0])
    axes[0].set_title(f"Axial Slice {axial_slider+1}/{img3d.shape[2]}")

    sagittal_view = np.rot90(img3d[sagittal_slider, :, :])
    axes[1].imshow(sagittal_view, cmap='gray')
    axes[1].set_aspect(aspects[1])
    axes[1].set_title(f"Sagittal Slice {sagittal_slider+1}/{img3d.shape[0]}")

    coronal_view = np.rot90(img3d[:, coronal_slider, :])
    axes[2].imshow(coronal_view, cmap='gray')
    axes[2].set_aspect(aspects[2])
    axes[2].set_title(f"Coronal Slice {coronal_slider+1}/{img3d.shape[1]}")

    for ax in axes:
        ax.axis('off')

    st.pyplot(fig)

    return axial_slider, sagittal_slider, coronal_slider

# Function to overlay RT Dose on the CT slices
def overlay_rt_dose_on_ct(temp_dir, img3d, axial_slider, sagittal_slider, coronal_slider, ax_aspect, sag_aspect, cor_aspect):
    rt_dose_file = None
    # Find RT Dose file in the temp directory
    for file_name in os.listdir(temp_dir):
        if "RTDOSE" in file_name.upper():
            rt_dose_file = os.path.join(temp_dir, file_name)
            break

    if rt_dose_file:
        # Read the RT Dose DICOM file
        dose_ds = pydicom.dcmread(rt_dose_file)

        # Rescale the RT Dose grid to match the CT image shape
        dose_array = dose_ds.pixel_array.astype(np.float32)
        dose_rescaled = zoom(dose_array, (
            img3d.shape[0] / dose_array.shape[0],
            img3d.shape[1] / dose_array.shape[1],
            img3d.shape[2] / dose_array.shape[2]
        ))

        # Rotate dose slices for proper alignment with the CT slices
        axial_dose_slice = dose_rescaled[:, :, axial_slider]
        sagittal_dose_slice = rotate(dose_rescaled[sagittal_slider, :, :], 90, reshape=False)
        coronal_dose_slice = rotate(dose_rescaled[:, coronal_slider, :], 90, reshape=False)

        # Plot CT slices with dose overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Axial slice with dose overlay
        axes[0].imshow(img3d[:, :, axial_slider], cmap='gray')
        dose_overlay_axial = axes[0].imshow(axial_dose_slice, cmap='jet', alpha=0.5)
        axes[0].set_aspect(ax_aspect)
        axes[0].set_title(f"Axial Slice {axial_slider+1}")
        plt.colorbar(dose_overlay_axial, ax=axes[0], label="Dose (Gy)")

        # Sagittal slice with dose overlay
        axes[1].imshow(np.rot90(img3d[sagittal_slider, :, :]), cmap='gray')
        dose_overlay_sagittal = axes[1].imshow(sagittal_dose_slice, cmap='jet', alpha=0.5)
        axes[1].set_aspect(sag_aspect)
        axes[1].set_title(f"Sagittal Slice {sagittal_slider+1}")
        plt.colorbar(dose_overlay_sagittal, ax=axes[1], label="Dose (Gy)")

        # Coronal slice with dose overlay
        axes[2].imshow(np.rot90(img3d[:, coronal_slider, :]), cmap='gray')
        dose_overlay_coronal = axes[2].imshow(coronal_dose_slice, cmap='jet', alpha=0.5)
        axes[2].set_aspect(cor_aspect)
        axes[2].set_title(f"Coronal Slice {coronal_slider+1}")
        plt.colorbar(dose_overlay_coronal, ax=axes[2], label="Dose (Gy)")

        # Remove axes labels
        for ax in axes:
            ax.axis('off')

        st.pyplot(fig)
    else:
        st.error("RT Dose file not found.")

# Function to list RT Plan beam information
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

# Main processing and display of CT slices
if uploaded_files:
    img3d, aspects = load_ct_slices(temp_dir)
    if img3d is not None:
        slices = display_ct_slices(img3d, aspects)
        overlay_rt_dose_on_ct(temp_dir, img3d, slices[0], slices[1], slices[2], aspects[0], aspects[1], aspects[2])
        load_rt_plan_and_extract_tags(temp_dir)
else:
    st.info("Please upload DICOM CT slices, RT Dose, RT Structure, and RT Plan files.")
