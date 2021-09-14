# SkinAid: A GAN-based Automatic Skin Lesion Monitoring Method for IoMT Frameworks

This is the official W-GAN code release of our paper:
#### Submitted for IEEE OITS International Conference on Information Technology, December-2021

###  Overview:
<img src="Images/design_overview.png" width="1000"/>

• Extracted the Region-of-interest of the Skin Lesions (Melanoma, Melanocytic Nevi, Benign Keratosis, Basal Cell Carcinoma, Actinic Keratosis, Vascular Lesions & Dermatofibroma) & pre-processing.<br>
<br>
• Enhanced the highly unbalanced & limited HAM10000 dataset by augmenting/generating synthetic Skin Lesion images using Wasserstein-GAN with Gradient penalty.<br>
<br>
• Trained our model to classify 7 types skin cancers/lesions using Transfer learning (ResNet, EfficientNet, DenseNet, MobileNet) and achieved a best accuracy of 92.2% with DenseNet-121.<br>
<br>
• Developed a prototype of an Android Application to capture real-time skin lesion image from smartphone camera to detect, classify & generate a preliminary analysis report, useful in rural or remote areas with limited healthcare access.<br>

### (a) Extracting the Region-of-interest:
<img src="Images/roi.png" width="1000"/>

<br>

### (b) Samples of Synthetic Images Generated using Wassetstein GAN:
<img src="synthetic_samples/2.png" width="200"/> <img src="synthetic_samples/3.png" width="200"/> <img src="synthetic_samples/1.png" width="200"/> <img src="synthetic_samples/4.png" width="200"/>

<br>

### (c) Training CNN models with Transfer Learning & Smartphone Deployment:
<img src="Images/mobile_app.png" width="600"/>
<br>
<img src="Images/cnn_acc_comparision.png" width="600"/>
