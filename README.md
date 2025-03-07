# Urban Green Space Mapping with U-Net

#### Project Overview
This project aims to map urban green spaces from Sentinel-2 satellite imagery using a U-Net deep learning model. The model is trained on labeled satellite images to segment and classify green areas in urban environments.

#### Dataset
The dataset consists of satellite images and corresponding ground truth masks indicating green space areas. The ground truth mask dataset used in training are from the [MULC](https://www.mdpi.com/2072-4292/12/12/1909) dataset.

#### Model Architecture
The model is based on a standard U-Net architecture, a convolutional neural network designed for image segmentation. It consists of:
- **Encoder**: Downsampling layers with convolutional and max-pooling operations
- **Bottleneck**: Bridge layer with high-level feature extraction
- **Decoder**: Upsampling layers with skip connections from the encoder

#### Training Details
- **Loss Function**: Binary Focal Cross-Entropy
- **Optimizer**: Adam
- **Batch Normalization**: Applied in each convolutional block
- **Dropout**: Used to prevent overfitting

### Example Prediction
![Urban Green Space Mapping Prediction Example](https://private-user-images.githubusercontent.com/166139720/420338124-bc73fa29-7b10-43d0-91a6-afa15adbe2c4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDEzNTM0MTgsIm5iZiI6MTc0MTM1MzExOCwicGF0aCI6Ii8xNjYxMzk3MjAvNDIwMzM4MTI0LWJjNzNmYTI5LTdiMTAtNDNkMC05MWE2LWFmYTE1YWRiZTJjNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMzA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDMwN1QxMzExNThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zYmVlODFjM2JhOTIzMGE1MGZhYjRiYzI5ZmMxZTVlMTEzMDdmMGQyNGVkNDQ1MjMwYTEyM2NjZDhjOTI5NzNjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9._WWuo3kasubArzQVDl2EuIogK4BARDcbtK4k4CV0BK8)

#### Results and Evaluation
The model achieves a mean IoU (Intersection over Union) score of 0.8641 on the test set. The segmented outputs closely match the ground truth, though some noise remains in certain cases.

###  Future Work
- Improve model performance with additional data augmentation
- Experiment with alternative pretrained backbones for the U-Net model
