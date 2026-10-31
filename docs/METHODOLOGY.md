# OILTRACE — Scientific Methodology, Physics, and Limitations
**Module**: Member 1 — AI-Based Oil Spill Detection  
**Platform**: European Space Agency (ESA) Sentinel-1 C-Band Synthetic Aperture Radar (SAR)  

---

## 1. Principles of Radar Remote Sensing of Marine Oil Slicks

### Radar Backscatter and Bragg Resonance
Synthetic Aperture Radar (SAR) sensors operate by emitting coherent microwave pulses (e.g., C-band, $\lambda \approx 5.6\text{ cm}$ on Sentinel-1) and measuring the backscattered energy $\sigma_0$ returned from the Earth's surface.

Over the open ocean, the primary scattering mechanism at intermediate incidence angles ($20^\circ \text{ to } 45^\circ$) is **Bragg scattering**. Microwaves scatter resonantly off short capillary and capillary-gravity surface waves whose wavelength $\lambda_w$ satisfies:
$$\lambda_w = \frac{\lambda}{2 \sin(\theta)}$$
where $\theta$ is the radar incidence angle. For C-band, resonant capillary waves are roughly $4\text{ to } 8\text{ cm}$ in length.

### The Damping Mechanism of Mineral Oil
When liquid hydrocarbons (crude or refined petroleum) form a surface film, they increase the surface tension gradient and viscoelastic damping of the water surface (Marangoni damping effect). This dampens short capillary-gravity waves, causing the sea surface to become geometrically smooth at the radar wavelength. 

Consequently, radar pulses undergo specular reflection away from the sensor rather than diffuse backscatter towards it. On the resulting SAR image, the oil slick appears as a **prominent low-backscatter dark formation** (typically $3\text{ dB to } 10\text{ dB}$ below the ambient sea clutter).

---

## 2. Detection Pipeline Architecture

### Phase 1: Speckle Noise Filtering in Linear Power Space
SAR images are intrinsically corrupted by **speckle noise**, a multiplicative interference pattern resulting from coherent summation of sub-resolution scatterers. Speckle intensity follows a Gamma distribution with unit coefficient of variation in single-look images.

Standard filtering must take place in **linear intensity power** ($I = \text{amplitude}^2$), **never in decibels (log scale)**:
$$\langle I \rangle = \frac{1}{N} \sum_{i=1}^N I_i$$
Averaging in log/decibel space causes a systematic negative bias ($-\gamma_{\text{Euler}} \approx -2.5\text{ dB}$ in single-look data), corrupting background radiometric estimation. OILTRACE uses a $5 \times 5$ multi-look boxcar filter operating in linear power space via integral image tables.

### Phase 2: Adaptive Backscatter Thresholding
In operational SAR surveillance, background sea clutter varies widely based on local wind speed and antenna incidence angles. OILTRACE combines a statistical offset with a dark percentile guard:
$$\tau = \min\left(\mu_{\text{sea}} - k \cdot \sigma_{\text{sea}}, \; P_{15}\right)$$
Pixels below $\tau$ are extracted as initial dark-formation candidates.

### Phase 3: Morphological Consolidation
To reject isolated speckle spikes and consolidate cohesive slicks, morphological operations are applied:
1. **Opening ($3 \times 3$)**: Binary erosion followed by dilation removes isolated single-pixel noise clusters.
2. **Closing ($3 \times 3$)**: Binary dilation followed by erosion bridges narrow fractures caused by internal wave shear.

### Phase 4: Feature Extraction and Look-Alike Discrimination
Connected components are analyzed for both radiometric and morphological features:
* **Contrast ($\Delta\text{dB}$)**: $10 \log_{10}(\mu_{\text{sea}} / \mu_{\text{slick}})$. Slicks typically require $\ge 2.5\text{ dB}$ to $4.0\text{ dB}$ contrast. Regions below $2.0\text{ dB}$ are dismissed as ambient noise.
* **Morphological Compactness**: $4\pi \cdot \text{Area} / \text{Perimeter}^2$. Slicks subjected to maritime drift and wind shear stretch into elongated shapes (compactness $< 0.40$), whereas rain cells and wind shadows tend toward circular or rounded geometries (compactness $> 0.65$).

---

## 3. Scientific Uncertainty and Look-Alike Phenomena

> [!WARNING]
> **A dark patch in a SAR image is NOT definitive chemical proof of mineral oil.**

Many natural atmospheric and oceanographic phenomena dampen capillary waves or reduce radar return in an identical visual manner:

| Phenomenon | Physical Cause | Distinction Mechanism in OILTRACE |
| :--- | :--- | :--- |
| **Biogenic Surfactants** | Natural algae, phytoplankton blooms, or fish oils forming mono-molecular films. | Typically thinner, lower contrast ($\Delta\text{dB} < 3.0$), and often aligned with ocean fronts. Flagged with look-alike uncertainty. |
| **Low-Wind Calms** | Wind speeds $< 2\text{ to } 3\text{ m/s}$ fail to generate capillary waves. | Large, diffuse boundaries, gentle gradients, weak contrast. Rejected by contrast filter or flagged. |
| **Rain Cells** | Heavy precipitation dampens surface waves and attenuates radar signals. | Often accompanied by bright convective edges. High circular compactness. |
| **Internal Waves** | Oceanic internal solitons modulating surface currents. | Alternating bright and dark parallel bands. Distinct periodic morphology. |
| **Upwelling / Cold Water** | Thermal stratifications modifying surface viscosity. | Broad regional extents without elongated tail structures. |

Because single-sensor SAR cannot chemically differentiate mineral crude from biogenic films, OILTRACE explicitly reports detected objects as **suspected candidates** with associated uncertainty flags, requiring Member 2's AIS vessel correlation and historical drift tracking to corroborate anthropogenic origins.

---

## 4. Dataset Provenance, Test Fixtures, and Limitations

### Source Data and Provenance
* **Original Statistical Source**: Garcia-INPE Sentinel-1 SAR Oil Spill Dataset (`Oil_Stats_manual_windows_8bits_800x600.csv`)
* **Covered Region**: Cantarell Oil Field Complex, Bay of Campeche, Gulf of Mexico ($19.378^\circ\text{N}, -92.035^\circ\text{W}$)
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Citation**: [10.5281/zenodo.22214144](https://doi.org/10.5281/zenodo.22214144)
* **Reference Satellite Scene**: ESA Copernicus Sentinel-1A (`S1A_IW_GRDH_1SDV_20200117T001541`), acquired 2020-01-17T00:15:41Z.

### Synthetic Test Fixture Disclosure
Because original raster images in the public Garcia-INPE GitHub tree are stored via Git LFS pointers and Zenodo downloads return HTTP 403 under automated retrieval, the sample files in `data/samples/` (`s1_cantarell_slick.png`, `s1_cantarell_reference.png`, `s1_clean_sea.png`, and `s1_lookalike_wind.png`) are **reproducible synthetic SAR test fixtures** generated by `data/generate_samples.py`. They are mathematically parameterized with the exact published background clutter ($\mu \approx 44, \sigma \approx 20$) and slick damping ($\mu \approx 18, \sigma \approx 5$) from row 1 of the Garcia-INPE Cantarell table.

### Georeferencing Bounds & Affine Mapping
The pixel-to-coordinate mapping uses a **nominal affine bounding box model** assuming $10\text{ m} \times 10\text{ m}$ pixel spacing (nominal Sentinel-1 IW GRDH resolution) centered on the Cantarell centroid coordinates ($19.378354^\circ\text{N}, -92.035142^\circ\text{W}$). It is an affine approximation rather than an embedded GeoTIFF ground control point transform.

### Evaluation Metrics Scope
The segmentation evaluation metrics (IoU, Dice, Precision, Recall) evaluate the classical extraction engine against the synthetic ground-truth fixture mask to verify that the multi-look filtering, adaptive thresholding, and morphological operations accurately recover dark patches. **These metrics verify code implementation correctness and must not be cited as real-world operational accuracy on uncurated satellite imagery.**

---

## 5. Real Sample Provenance — MKLab-ITI Oil Spill Detection Dataset

### Dataset Identification
* **Dataset**: Oil Spill Detection Dataset (SAR Semantic Segmentation)
* **Authors**: Krestenitis M., Orfanidis G., Ioannidis K., Avgerinakis K., Vrochidis S., Kompatsiaris I.
* **Laboratory**: Multimedia Knowledge and Social Media Analytics Laboratory (MKLab), CERTH-ITI, Greece
* **Publication**: Krestenitis et al. (2019). "Oil Spill Identification from Satellite Images Using Deep Neural Networks." *Remote Sensing*, 11(15), 1762. DOI: [10.3390/rs11151762](https://doi.org/10.3390/rs11151762)
* **Dataset Portal**: https://m4d.iti.gr/oil-spill-detection-dataset

### Dataset Characteristics
The dataset contains **1,112 tiles** of Sentinel-1 SAR imagery at a standard tile size of **1250 × 650 pixels**. Each tile is annotated with a 5-class semantic mask:
1. Sea surface (0), Oil spill (1), Look-alike (2), Ship (3), Land (4)

Tiles were derived from ESA Copernicus Sentinel-1 C-band VV-polarization GRD products. The two JPEG samples in `data/real_samples/` (`sentinel1_mklab_sample_01.jpg`, `sentinel1_mklab_sample_02.jpg`) are tiles from this dataset, identifiable by their exact 1250 × 650 pixel dimensions and characteristic SAR speckle texture.

### Access Terms & License

> [!WARNING]
> The MKLab-ITI dataset is **NOT** under an open CC BY license. It is distributed **access-on-request for non-commercial research and educational use only** via the CERTH-ITI portal. Redistribution or public hosting of raw image tiles is prohibited by the dataset terms of use. See `data/real_samples/PROVENANCE.md` for the full citation and terms.

### What Is Unavailable for These Tiles
* **No embedded geographic metadata**: JPEG tiles contain no GPS EXIF, scene center, or GeoTIFF GCPs. Pixel-to-coordinate mapping is not possible and is explicitly **not performed**.
* **No acquisition timestamp**: Overpass time is not encoded in tile filenames.
* **No reference mask**: Paired segmentation masks are part of the full dataset distribution and were not included. Accordingly, **no evaluation metrics** (IoU, Dice, Precision, Recall) are reported for these tiles.

