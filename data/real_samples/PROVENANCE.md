# Data Provenance — `data/real_samples/`

## Image Files

| Filename | Size | Dimensions |
| :--- | :--- | :--- |
| `sentinel1_mklab_sample_01.jpg` | ~184 KB | 1250 × 650 px |
| `sentinel1_mklab_sample_02.jpg` | ~225 KB | 1250 × 650 px |

---

## Source Dataset

**Dataset Name**: Oil Spill Detection Dataset (SAR Semantic Segmentation)  
**Authors / Lab**: Krestenitis et al., Multimedia Knowledge and Social Media Analytics Laboratory (MKLab), Centre for Research and Technology Hellas — Information Technologies Institute (CERTH-ITI)  
**Publication**: Krestenitis M., Orfanidis G., Ioannidis K., Avgerinakis K., Vrochidis S., Kompatsiaris I. (2019). "Oil Spill Identification from Satellite Images Using Deep Neural Networks." *Remote Sensing*, 11(15), 1762. https://doi.org/10.3390/rs11151762  
**Dataset Portal**: https://m4d.iti.gr/oil-spill-detection-dataset  
**GitHub (code)**: https://github.com/MKLab-ITI/oil-spill-detection

---

## Access Terms & License

> [!WARNING]
> **These images are NOT under a standard open-source or Creative Commons license.**

The MKLab Oil Spill Detection Dataset is distributed **on request for academic and non-commercial research purposes only**. Raw image files must be obtained directly from the authors/lab via the official dataset portal request form. Redistribution, commercial use, or public hosting of the raw image files is generally prohibited under the dataset's terms of use.

**These sample files were obtained for use in the OILTRACE Galuxium Nexus hackathon as a non-commercial research and educational demonstration only.**

---

## Image Characteristics

- **Sensor**: ESA Copernicus Sentinel-1 C-Band SAR
- **Polarization**: VV
- **Processing Level**: Detected GRD (Ground Range Detected), amplitude
- **Standard Tile Size**: 1250 × 650 pixels
- **Format**: JPEG (lossy compression applied; original data is 16-bit GeoTIFF)
- **Classes annotated in dataset**: Sea surface (0), Oil spill (1), Look-alike (2), Ship (3), Land (4)

---

## What Is Not Available in These Files

- **No embedded EXIF or geographic metadata**: The JPEG tiles contain no embedded GPS, scene center, or GeoTIFF ground control points. Pixel-to-coordinate mapping is **not possible** from the file alone.
- **No acquisition timestamp**: The exact overpass time is not encoded in the filename or file metadata of these tiles.
- **No reference mask**: The paired semantic segmentation masks are part of the full dataset distribution and were not included here. Accordingly, **no IoU, Dice, Precision, or Recall metrics** are reported against these images.

---

## Distinction from Synthetic Fixtures

The files in `data/samples/` (e.g., `s1_cantarell_slick.png`) are **synthetically generated** test fixtures parameterized from Garcia-INPE Cantarell dataset statistics (CC BY 4.0). They are **completely separate** from the real MKLab-derived samples in this directory.
