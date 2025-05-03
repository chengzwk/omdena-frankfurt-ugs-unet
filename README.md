# 🌿 Urban Green Space Mapping with U-Net

## Project Overview

This project focuses on mapping urban green spaces using Sentinel-2 satellite imagery and U-Net-based deep learning models. We investigate the effectiveness of different input band combinations and model architectures for semantic segmentation of green spaces in urban environments in the city of Frankfurt, Germany.


## Dataset

The primary goal of this project is to predict urban green spaces in **Frankfurt**, Germany. To generate high-quality ground truth, we annotated 13 high-resolution aerial photographs of Frankfurt using **SAM-2** with additional manual corrections. Due to the limited number of annotated images, Frankfurt data was reserved exclusively for final evaluation and not used during model development.

For model training and validation, we used the **MULC dataset** ([Reference](https://www.mdpi.com/2072-4292/12/12/1909)), focusing on the **Virginia Beach/Williamsburg (VBWVA)** and **St. Louis** regions. The VBWVA region, being more similar to Frankfurt in landscape and urban structure, was used for training, validation, and testing splits. All available St. Louis images were included in the training set to increase diversity and improve model generalization.


## Model Architectures

### U-Net (from scratch)
A standard U-Net model trained without a pretrained backbone, tested with multiple band combinations.

### U-Net with ResNet-50 Backbone
A U-Net model using a ResNet-50 encoder pretrained on ImageNet. This architecture is limited to three-channel input combinations.


## Data Pre-processing
- **Compute band index**: Compute NDVI and NDWI
- **Normalization**: Normalize each band to [0,1]
- **Convert to binary**: Convert the mask from fractional (fraction of each of the 5 classes) to binary (0 for non-vegetation, 1 for vegetation)


## Training Configuration

- **Optimizer**: AdamW with learning_rate=1e-3, weight_decay=5e-5
- **Loss Function**: Binary Focal Cross-Entropy
- **Batch Normalization**: Applied in all convolutional blocks
- **Dropout**: Not included
- **Data Augmentation**: Applied to both images and masks using random rotation (±45°), width and height shifts (up to 20%), zoom (up to 20%), horizontal and vertical flips. All transformations use `reflect` fill mode. For masks, a post-processing step binarizes pixel values using a 0.5 threshold.
- **Learning Rate Scheduler**: Exponentially decays the learning rate by a factor of `exp(-0.1)` every 10 epochs starting from epoch 200. 


## Input Band Combinations

| Band Combination         | Applicable Models              |
|--------------------------|--------------------------------|
| Red-Green-Blue           | U-Net, U-Net with ResNet-50    |
| Red-Green-NIR            | U-Net, U-Net with ResNet-50    |
| NDVI-Red-NIR             | U-Net, U-Net with ResNet-50    |
| NDWI-Red-NIR             | U-Net, U-Net with ResNet-50    |
| Red-Green-Blue-NIR       | U-Net only                     |


## Results and Evaluation

Model performance was evaluated on both the VBWVA test region and the independent Frankfurt dataset. Key metrics include overall accuracy (OA), F1-score, and mean Intersection over Union (IoU). The metrics for best-performing models on both architectures are:

| Model                     | Dataset     | OA     | F1-score | Mean IoU |
|--------------------------|-------------|--------|----------|----------|
| U-Net (from scratch)     | VBWVA       | 0.8639 | 0.8921   | 0.7604   |
| U-Net (from scratch)     | Frankfurt   | 0.7632 | 0.7686   | 0.6171   |
| U-Net + ResNet-50        | VBWVA       | 0.8656 | 0.8891   | 0.7631   |
| U-Net + ResNet-50        | Frankfurt   | 0.7727 | 0.7701   | 0.6296   |

Both models demonstrated strong performance on the VBWVA dataset and showed reasonable generalization to the Frankfurt dataset, despite domain differences, the difference in annotation across the two datasets and no training exposure to Frankfurt imagery.


### Comparison of model performance across band combinations
- ![U-Net - VBWVA](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/model_performance_VBWVA_testset.png)    
- ![U-Net - Frankfurt](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/model_performance_Frankfurt.png)  

### Reports and Notebooks

- [Evaluation Report – U-Net from Scratch](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/models_evaluation_report_metrics.ipynb)
- [Training Notebook – U-Net from Scratch](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/unet-from-scratch_report.ipynb)
- [Evaluation Report – U-Net with ResNet-50 Backbone](https://dagshub.com/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/src/main/TeamBC/Unet_Resnet50/unet_resnet50_models_evaluation_report_metrics.ipynb)
- [Training Notebook – U-Net with ResNet-50 Backbone](https://dagshub.com/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/src/main/TeamBC/Unet_Resnet50/unet-resnet50.ipynb)

## Example Predictions

### Model prediction on VBWVA images (U-Net from Scratch and U-Net with ResNet-50 Backbone)
- ![Example - U-Net - VBWVA](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/Predictions%20on%20VBWVA%20test%20images.png)
- ![Example - ResNet-50 - VBWVA](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_resnet50/Predictions%20on%20VBWVA%20test%20images.png)

### Model prediction on Frankfurt images (U-Net from Scratch and U-Net with ResNet-50 Backbone)
- ![Example - U-Net - Frankfurt](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/raw/8a4fbfe6082771e2468b05744a0777872b80c5ff/report/unet_from_scratch/Predictions%20on%20Frankfurt%20images.png)

- ![Example - ResNet-50 - Frankfurt](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/raw/8a4fbfe6082771e2468b05744a0777872b80c5ff/report/unet_resnet50/Predictions%20on%20Frankfurt%20images.png)

### Model Prediction vs. Ground Truth on Frankfurt Images

- [U-Net from Scratch](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_from_scratch/Prediction%20vs%20Ground%20Truth%20on%20Frankfurt%20images.png)
- [U-Net with ResNet-50 Backbone](https://dagshub.com/chengzwk/omdena-frankfurt-ugs-unet/src/main/report/unet_resnet50/Prediction%20vs%20Ground%20Truth%20on%20Frankfurt%20images.png)

## Future Work

- Perform more in-depth error analysis on Frankfurt prediction results
- Test alternative pretrained encoder backbones

## Contact

For questions or feedback, please feel free to reach out via the project page or submit an issue.

