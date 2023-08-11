# geoinf
Geoinformed project code

# dataset preparation
1. Kaartbladen with size of four has been used in this application, for details of kaartbladenversnijdingen, see: https://download.vlaanderen.be/product/111-kaartbladversnijdingenngiklassiekereeks
2. Run generate_kaartbladen.py to generate train dataset, and run *_test.py *_val.py repectively for test and validation dataset.
3. Run compute_cloud_masks4.py to generate cloud masks for dataset, changing kaartbladen numbers for test and validation sets.

# model training
1. train the vanilla unet model with train_4.py
2. test the best trained model settings with val_4.py
