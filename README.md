# Data augmentation for skin lesion detection

**BSc Computer Engineering Thesis Project**

## Abstract

Deep learning models show remarkable results in the automated analysis of skin lesions. However, these models require a considerable amount of data and the availability of medical images of this type is often limited. The creation of artificial examples can be the solution to expand the amount of images available to train the model.

This work, fine-tuning the pre-trained CNN AlexNet, analyzes the impact of different data augmentation methods for the classification of skin lesions. Implemented methods include geometric transformations and chromatic variations.

Results obtained confirm the importance of adding artificial examples in the training phase. The best obtained data augmentation scenario increased the model accuracy from 0.8297 to 0.8609, based on the correct classification of skin lesions.

## Dataset

HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (10015 images, size of 450*600 pixels). Download link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.

Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: 
1. akiec : Actinic keratoses and intraepithelial carcinoma / Bowen's disease;
2. bcc   : basal cell carcinoma;
3. bkl   : benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses);
4. df    : dermatofibroma;
5. mel   : melanoma;
6. nv    : melanocytic nevi;
7. vasc  : vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage).

## Files

- *SkinAlexNetDataAug.m:* main function that calls data augmentation methods from "myImageDataAugmenter.m" file, fine-tune the pre-trained AlexNet and test results.
- *myImageDataAugmenter.m:* contains all proposed data augmentation methods.
- *Skin_JPGtoMAT.m:* script to convert the dataset folder (containing 10015 file .jpg) to a single file .mat (name used in the code: *skin_dataset.mat*).
