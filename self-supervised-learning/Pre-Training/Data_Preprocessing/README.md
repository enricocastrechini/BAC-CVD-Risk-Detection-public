# Data Preprocessing

This folder contains the upstream DICOM-to-2D-PNG utility `LIDC_3DDICOM_to_2Dpng.py`. It was inherited from the original CT-focused self-supervised-learning project and is not required when the mammography images have already been converted and preprocessed by the main project pipeline.

Use it only with authorized DICOM data. Before running, inspect the paths in the script and direct the output to a local, ignored directory. The conversion does not apply CT windowing because the original use case pretrains on all available slice information; this choice is not automatically appropriate for mammography.

The dependencies are listed in `requirements.txt`. Keep the input data, converted images and any metadata outside the public repository.
