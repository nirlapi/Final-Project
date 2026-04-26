# Thermal Imaging and CNN-based Machine Learning for Knee Pain Assessment

## Project Status

This project focuses on the development of a research-oriented deep learning pipeline for classifying clinically significant knee pain using infrared thermal images. The project is based on thermal knee images collected from soldiers across several visits, together with tabular clinical data containing patient identifiers, knee side, visit information, and pain-related labels.

The overall goal is to build a supervised machine learning pipeline in which thermal knee images are preprocessed, matched with the relevant clinical labels, split correctly at the patient level, and then used to train and evaluate a Convolutional Neural Network (CNN) for binary pain classification.

At the current stage, the project is in the middle of the preprocessing and dataset-construction phase. The raw images were extracted from the medical system, and the relevant anatomical regions were manually segmented. In addition, several preprocessing and validation steps have already been implemented: loading all image files, resizing the images to a uniform format, checking consistency between the image-name prefix and the patient number (PN) in the Excel metadata file, attaching the appropriate label from the Excel file to each image, and identifying images that do not currently have a matching label.

These steps are essential because the project combines visual data with tabular clinical information. Therefore, before model training, it is necessary to ensure that every image is correctly linked to the relevant subject, visit, knee side, anatomical region, and pain label. This also helps prevent incorrect supervision and reduces the risk of data leakage or mislabeled samples.

The next stages of the project include splitting the dataset by patients into training, validation, and test sets, so that images from the same patient do not appear in more than one subset. After that, data augmentation will be applied only to the training set in order to increase variability while preserving the validity of the validation and test sets. The final image arrays and label arrays will then be constructed and used as input for the CNN model.

After completing the preprocessing pipeline, the next major phase will be model development. This will include building the CNN architecture, training the model using the training set, tuning and monitoring performance on the validation set, and evaluating the final model on the test set. The evaluation will include standard classification metrics such as accuracy, ROC-AUC, sensitivity, and specificity, with particular attention to the clinical meaning of false negative predictions.

The final stage of the project will include analysis of the results, discussion of model performance, identification of limitations, and formulation of conclusions regarding the ability to classify knee pain from thermal imaging data.