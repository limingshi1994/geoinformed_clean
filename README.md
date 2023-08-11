# GeoInformed

The following sequence of actions should be performed in order to prepare the data

## Create a conda environment from the requirements and activate the environment

    conda create --name myenv --file requirements.txt
    conda activate myenv

## Add 'BVM_labeled.zip' to the 'resources' directory

## To download the data, run the following command:

    python src/generate_kaartbladen.py --kaartbladen 2 4 --temporal-extent 2022-03-05 2022-03-10 --out-dir "generated_data" --gt-file resources/BVM_labeled.zip --kaartbladen-file resources/Kbl.shp

NOTE: The command example generates the ground truth and downloads sattelite images for kaartbladen 2 and 4 (any kaartbladen between 1 and 43 can be listed), within the temporal extent between 2022-03-05 and 2022-03-10, and creates a directory 'generated_data' in which the data is stored (any other path may be given)

## To precompute the cloud masks, run the following command:

    python src/compute_cloud_masks.py --kaartbladen 2 4 --years 2022 --months 3 --root-dir "generated_data"

NOTE: The command example covers the previous temporal extent of 2022-03-05 and 2022-03-10 and kaartbladen 2 and 4

## (OPTIONAL) To compute the normalization, run the following command (not necessary, some normalizations are already computed and stored in src/utils/constants):

    python src/compute_normalization.py --kaartbladen 2 4 --years 2022 --months 3 --root-dir "generated_data" --num-samples 100

CAUTION: Using more than 100 samples may fill up the RAM (100 samples takes up about 8 Gb)

## Finally, train and test dataloaders can be tested by running the following scripts:

    python src/train_dataloading.py --kaartbladen 2 4 --years 2022 --months 3 --root-dir "generated_data" --patch-size 256 --valid-threshold 0.3
    python src/test_dataloading.py --kaartbladen 2 4 --years 2022 --months 3 --root-dir "generated_data" --patch-size 256 --patch-offset 128 --valid-threshold 0.3
