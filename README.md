# Diagnosis of retinal diseases driven by Mamba multi-modal CLIP auxiliary fusion network

# Data Preparation！
1.Download dataset:
https://ieee-dataport.org/open-access/octa-500.

2.Prepare the dataset：

Create a folder named OCTA-300

Merge octa_3m_oct_mart1 to octa_3m_oct_mart4 into the OCTA-300 \ OCT folder

Copy the octa-500_ground_truth \ OCTA-500 \ OCTA_3M \ Project Maps \ OCT (FULL) folder to OCTA-300 \

Copy octa-500_ground_truth \ OCTA-500 \ OCTA3M \ Text labels.xlsx to OCTA-300 \.

The processing of datasets in the dataprocess.py

lib：

The lib package is some data reading code！

# Code run！
Bscan is the code execution for Bscan slicing.

OCT is the code for OCT branch.

CFP is the code for CFP branch.

Multi is the code for Multi branch.

# Citation！
If you find this work helpful for your project, please consider citing the following paper:

