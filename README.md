# Segmentation
Unet &amp; ResUnet Architecture for Medical Image analysis

Myeloproliferative disorders are a group of hematopoietic neoplasms characterized by clonal proliferation of any of the hematopoetic lineage like Erythroid-Polycythemia vera, Myeloid-CML, megakaryocytic/platelet proliferation-Essential thrombocythemia, stomal component-Primary Myelofibrosis. It is well known that certain morphologic features of megakaryocyte like loose and dense clustering of megakaryocytes with cloud like nuclei are seen in Polycythemia vera, dense clustering of megakaryocytes with stag horn nuclei is seen in Myelofibrosis, large size megakaryocytes with stag horn multinucleate nuclei are seen in Essential thrombocythemia along with secondary features like sinusoidal dilatation and extra medullary hematopoiesis. Dwarf megakaryocytes is a feature of chronic myeloid leukemia. Microscopic studies and morphologic evaluation becomes mandatory in all these cases to come for a conclusive diagnosis.

We are trying to exploit the Morphological features using deep learning techniques and narrow down on the Region of Interest and eventually perform a semantic segmentation task. By performing an Image segmentation we will be able to classify each and every pixel into its category. To perform this task we have used two Deep learning Networks UNet and ResUnet++.

The code is developed with the architecture of UNet and ResUnet++, also has a custom training script.

We also developed a GUI version of the same and had given it to the doctors for further review. This GUI helps them upload an image and obtain a semantic mask as output which further gives us a classification output too.

We at NITK had colloborated this project with Department of Pathology, Kasturba Medical College,Mangalore, MAHE, Manipal. As the data is sensitive we are unable to disclose the data, but the segmentation outputs where really appreciable by the Doctors at KMC.

UNet Paper - https://arxiv.org/abs/1505.04597
ResUNet Paper - https://arxiv.org/abs/1904.00592
