import numpy as np
import javabridge
import bioformats
import os
import xml.etree.ElementTree as ET


def read_oir(file_path) -> tuple[np.ndarray, bioformats.OMEXML]:
    """
    Method to read an Evident .oir file and return the image data as numpy array and metadata as xml.

    Example usage:
    img, metadata = read_oir(file_path)

    Args:
        file_path (str): Path to the Evident image file.

    Returns:
        image (np.ndarray): Acquired image data.
        metadata (OMEXML): metadata
    """

    if not javabridge.get_env():
        print("Starting JVM...")
        javabridge.start_vm(class_path=bioformats.JARS)
        javabridge.attach()

    if not javabridge.get_env():
        raise RuntimeError(
            "JVM is not running. Please start it before calling this function."
        )

    metadata = bioformats.get_omexml_metadata(file_path)
    meta_data = bioformats.OMEXML(
        metadata
    )  # takes that raw XML string and parses it into a Python object that’s easier to work with. The OMEXML class understands the OME-XML schema and gives you structured access.

    # Extract dimensions
    size_x = meta_data.image().Pixels.SizeX
    size_y = meta_data.image().Pixels.SizeY
    size_z = meta_data.image().Pixels.SizeZ
    size_c = meta_data.image().Pixels.SizeC
    size_t = meta_data.image().Pixels.SizeT

    # Initialize array in (x, y, z, time, channel) order
    image_stack = np.zeros((size_x, size_y, size_z, size_t, size_c), dtype=np.uint16)

    with bioformats.ImageReader(file_path) as reader:
        # Read and fill the array
        for t in range(size_t):
            for c in range(size_c):
                for z in range(size_z):
                    img = reader.read(c=c, z=z, t=t, rescale=False)
                    image_stack[:, :, z, t, c] = img

    return image_stack, meta_data


def convert_oir_folder(input_folder, force=False):
    """
    Convert all .oir files in the input folder to NumPy arrays and save them.

    Args:
        input_folder (str): Path to the folder containing .oir files.
        force (bool): If True, overwrite existing files. Default False.
    """

    output_folder = os.path.join(input_folder, "oir2numpy")

    # Folder handling
    if os.path.exists(output_folder):
        if not force:
            raise FileExistsError(
                f"Output folder '{output_folder}' already exists. "
                "Use force=True or remove it."
            )
    os.makedirs(output_folder, exist_ok=True)

    # Loop over files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".oir"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")

            try:
                img, metadata = read_oir(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            base_name = os.path.splitext(filename)[0]
            numpy_file_name = os.path.join(output_folder, base_name + ".npz")

            # Correct overwrite logic
            if os.path.exists(numpy_file_name) and not force:
                print(f"Skipping existing file: {numpy_file_name}")
                continue

            np.savez_compressed(numpy_file_name, image=img, metadata=metadata)
            print(f"Saved: {numpy_file_name}")

    print("Conversion complete!")


def read_metadata(metadata_str):
    """
    Parse OME-XML metadata string into a structured dictionary of imaging parameters.

    Extracts key microscopy metadata from an OME-XML (Open Microscopy Environment)
    formatted string, including instrument specifications, image properties, pixel
    information, and Z-stack positioning data.

    Parameters
    ----------
    metadata_str : str
        OME-XML metadata string (typically from TIFF image metadata).

    Returns
    -------
    dict
        Dictionary containing extracted metadata with the following keys:

        - 'lasers': list of dicts
            Laser attributes from the Instrument element.
        - 'detectors': list of dicts
            Detector attributes from the Instrument element.
        - 'objectives': list of dicts
            Objective attributes from the Instrument element.
        - 'image': dict
            Image attributes including 'AcquisitionDate' if available.
        - 'pixels': dict
            Pixel attributes with additional computed fields:

            - 'FocusDistance': str
                Calculated Z-stack range with units (e.g., "100.5 µm").
            - 'FocusDistanceRaw': float
                Numerical Z-stack range without units.
            - 'ZPositions': list of float
                Sorted unique Z positions from Plane elements.

    Notes
    -----
    - Uses OME schema version 2016-06.
    - Z-stack focus distance is calculated as the difference between maximum
      and minimum Z positions across all planes.
    - Duplicate Z positions are removed before returning.
    - Returns only information present in the metadata; missing elements are skipped.
    """
    root = ET.fromstring(str(metadata_str))

    # Define namespace
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    metadata_dict = {}

    # Extract Instrument information
    instrument = root.find(".//ome:Instrument", ns)
    if instrument is not None:
        lasers = instrument.findall("ome:Laser", ns)
        metadata_dict["lasers"] = [laser.attrib for laser in lasers]

        detectors = instrument.findall("ome:Detector", ns)
        metadata_dict["detectors"] = [detector.attrib for detector in detectors]

        objectives = instrument.findall("ome:Objective", ns)
        metadata_dict["objectives"] = [objective.attrib for objective in objectives]

    # Extract Image information
    image = root.find(".//ome:Image", ns)
    if image is not None:
        metadata_dict["image"] = image.attrib
        acq_date = image.findtext("ome:AcquisitionDate", namespaces=ns)
        if acq_date:
            metadata_dict["image"]["AcquisitionDate"] = acq_date

    # Extract Pixels information
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is not None:
        metadata_dict["pixels"] = pixels.attrib

        # Extract Z positions from Plane elements
        planes = pixels.findall("ome:Plane", ns)
        if len(planes) > 1:
            z_positions = []
            for plane in planes:
                pos_z = plane.attrib.get("PositionZ")
                if pos_z:
                    z_positions.append(float(pos_z))

            if z_positions:
                z_positions.sort()
                focus_distance = z_positions[-1] - z_positions[0]
                z_unit = planes[0].attrib.get("PositionZUnit", "µm")
                metadata_dict["pixels"]["FocusDistance"] = f"{focus_distance} {z_unit}"
                metadata_dict["pixels"]["FocusDistanceRaw"] = focus_distance
                metadata_dict["pixels"]["ZPositions"] = list(
                    set(z_positions)
                )  # Remove duplicates

    return metadata_dict
