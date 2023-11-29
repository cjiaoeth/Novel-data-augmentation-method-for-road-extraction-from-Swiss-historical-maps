from osgeo import ogr, gdal
import osgeo, os


import utils
from segmentation_manager import SegmentationManager
from data_manager import DataManager


sm = SegmentationManager("D:/Catherine/doctoral studies/road extraction/test/20220106/instances/", "F:/DL/models/")
dummy_instance = sm.get_instance("instance_road_siegfried-small-batch-multi-4")


target_path = "D:/Catherine/doctoral studies/road extraction/test/20220106/predictions/"
base_path = "D:/Catherine/doctoral studies/road extraction/test/sheets/"
sheets = ["rgb_TA_017_1940.tif", "rgb_TA_199_1941.tif", "rgb_TA_219_1944.tif", "rgb_TA_385_1941.tif"]
# sheets = ["rgb_TA_072_1880.tif"]


for sheet in sheets:
    sheet_path = base_path + sheet
    prediction_path = target_path + sheet

    dummy_instance.predict_sheet(sheet_path, prediction_path, batch_size = 20, resolution = 1.25, padding = 32, img_size=64, upsample=False)





