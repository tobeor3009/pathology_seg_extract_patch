
---

# Pathology Data Analysis Repository

This repository provides a step-by-step guide to working with pathology data, including:

1. **Extracting Tissue Mask from Images**  
   Refer to the notebook: `code/1_Extract_Tissue_Mask.ipynb`

2. **Generating Pathology Segmentation Patches (Glomerulus)**  
   Utilizing the tissue mask to extract segmentation patches.  
   Refer to the notebook: `code/2_Extract_Glomerulus_Patch.ipynb`

### Data Preparation and Results

- Place the downloaded dataset in `data/0_source_data/hubmap-kidney-segmentation` and process it to obtain the same results as described in this repository.
- The complete tissue mask results can be found in the directory: `data/1_tissue_mask`.
- The extracted segmentation patches (e.g., glomerulus patches) are stored in the directory: data/2_extract_patch.
  
<p align="center">
  <img src="code/example_images/0486052bb.png" alt="WSI Example" width="45%">
  <img src="code/example_images/0486052bb_tissue_mask.png" alt="Tissue Mask Example" width="45%">
</p>

<p align="center">
  <b>Figure 1:</b> Original WSI (left) and Tissue Mask (right).
</p>

<p align="center">
  <img src="data/2_extract_patch/0486052bb.tiff/negative/0000/000097/image.png" alt="Image Patch Example" width="30%">
  <img src="data/2_extract_patch/0486052bb.tiff/negative/0000/000097/mask.png" alt="Mask Patch Example" width="30%">
  <img src="data/2_extract_patch/0486052bb.tiff/negative/0000/000097/tissue.png" alt="Tissue Patch Example" width="30%">
</p>

<p align="center">
  <b>Figure 2:</b> Negative Image Patch (left), Mask Patch (center), Tissue Patch (right).
</p>

<p align="center">
  <img src="data/2_extract_patch/0486052bb.tiff/positive/0000/000096/image.png" alt="Image Patch Example" width="30%">
  <img src="data/2_extract_patch/0486052bb.tiff/positive/0000/000096/mask.png" alt="Mask Patch Example" width="30%">
  <img src="data/2_extract_patch/0486052bb.tiff/positive/0000/000096/tissue.png" alt="Tissue Patch Example" width="30%">
</p>

<p align="center">
  <b>Figure 3:</b> Positive Image Patch (left), Mask Patch (center), Tissue Patch (right).
</p>

### Data Source

The dataset used in this project is sourced from the Kaggle competition:  
[HuBMAP - Kidney Segmentation](https://www.kaggle.com/competitions/hubmap-kidney-segmentation)

### Future Work

We are currently preparing a research paper on **Segmentation Loss** based on the methods demonstrated in this repository.

### Contact

If you have any questions or suggestions, feel free to reach out via:  
📧 **Email**: tobeor3009@gmail.com  
💬 **GitHub Issues**

--- 

Let me know if you need further refinements! 😊