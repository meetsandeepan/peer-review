import streamlit as st
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import tempfile

# Function to load DICOM series using SimpleITK
def load_dicom_series(dicom_files):
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(dicom_files)
    image = series_reader.Execute()
    return sitk.GetArrayFromImage(image)

# Read metadata using SimpleITK
def read_dicom_metadata(dicom_files):
    ds = sitk.ReadImage(dicom_files[0])  # Reading metadata from the first file
    metadata = {tag: ds.GetMetaData(tag) for tag in ds.GetMetaDataKeys()}
    return metadata

# Function to display CT slices in 3 axes
def display_ct_slices(image):
    axial_slice = image[image.shape[0] // 2, :, :]
    sagittal_slice = image[:, image.shape[1] // 2, :]
    coronal_slice = image[:, :, image.shape[2] // 2]
    st.image(axial_slice, caption="Axial View", use_column_width=True, clamp=True, channels="GRAY")
    st.image(sagittal_slice, caption="Sagittal View", use_column_width=True, clamp=True, channels="GRAY")
    st.image(coronal_slice, caption="Coronal View", use_column_width=True, clamp=True, channels="GRAY")

# Display metadata in a table
def display_metadata(metadata):
    df = pd.DataFrame(list(metadata.items()), columns=["Tag", "Value"])
    st.dataframe(df)

# Streamlit app
st.title("DICOM CT Viewer with Metadata")

# Drag and drop file uploader
uploaded_files = st.file_uploader("Upload DICOM Files", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to a temporary directory
    temp_dir = tempfile.mkdtemp()
    dicom_files = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        dicom_files.append(file_path)

    if dicom_files:
        st.write("Loading DICOM series...")

        # Load DICOM images
        image = load_dicom_series(dicom_files)

        # Display slices
        display_ct_slices(image)

        # Read and display metadata
        metadata = read_dicom_metadata(dicom_files)
        display_metadata(metadata)
