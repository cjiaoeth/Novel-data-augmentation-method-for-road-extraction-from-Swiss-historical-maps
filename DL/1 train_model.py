from osgeo import ogr, gdal
import osgeo, os


import utils
from segmentation_manager import SegmentationManager
from data_manager import DataManager


sm = SegmentationManager("D:/Catherine/doctoral studies/road extraction/test/20220106/instances/", "F:/DL/models/")
sm.create_instance("instance_road_siegfried-small-batch-multi-4", "U-Net-Small-Batch-Multi", {"num_models" : 1, "num_base_filters" : 16}, ["roads"], True)

dummy_instance = sm.get_instance("instance_road_siegfried-small-batch-multi-4")


training_points = utils.NDFileManager.file("D:/Catherine/doctoral studies/road extraction/test/20220106/training_points_aggregated.nd")
vector_data = utils.NDFileManager.file("D:/Catherine/doctoral studies/road extraction/test/20220106/training_datasets.nd")
sheet_data = utils.NDFileManager.file("D:/Catherine/doctoral studies/road extraction/test/20220106/training_sheets.nd")

training_data_location = dummy_instance.prepare_training_data(training_points, sheet_data, vector_data)

data_augmentation = True

if data_augmentation:
    # buffer_raster = 
    dummy_instance.train_DA(training_data_location, epochs=100, early_stopping_patience=50, batch_size=64, upscale=False)
else:
    dummy_instance.train(training_data_location, epochs=100, early_stopping_patience=50, batch_size=64, upscale=False)




