# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:21:28 2021

@author: delladmin
"""

import os
import numpy as np
import random
from matplotlib import pyplot
from numpy import expand_dims
import fiona
import gdal, ogr, osr
import sys
import matplotlib.pyplot as plt


def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

    raster=r'somerasterfile.tif'
    ds=gdal.Open(raster)
    
    gt=ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext=GetExtent(gt,cols,rows)
    
    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    #tgt_srs=osr.SpatialReference()
    #tgt_srs.ImportFromEPSG(4326)
    tgt_srs = src_srs.CloneGeogCS()
    
    geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)
    return geo_ext


def zonal_stats(input_value_raster, input_zone_polygon):

    # Open data
    raster = gdal.Open(input_value_raster)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # reproject geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
            
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    # mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))

    # calculate mean of zonal raster
    return np.mean(zoneraster)


def rasterize(in_shp,out_tif):
    # Define pixel_size and NoData value of new raster
    pixel_size = 1.25
    NoData_value = -9999
    
    # Open the data source and read in the extent
    source_ds = ogr.Open(in_shp)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    
    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(out_tif, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0])

def rotate90(road_array,img_arr_copy):
    rotate90_arr1 = np.rot90(road_array[0],1)
    rotate90_arr2 = np.rot90(road_array[1],1)
    rotate90_arr3 = np.rot90(road_array[2],1)
    rotate90_arr4 = np.rot90(road_array[3],1)
    rotate90_arr = []
    rotate90_arr.append(rotate90_arr1)
    rotate90_arr.append(rotate90_arr2)
    rotate90_arr.append(rotate90_arr3)
    rotate90_arr.append(rotate90_arr4)
    rotate90_arr = np.array(rotate90_arr)
    
    # if len(rotate90_arr.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    rotate90_arr_copy = img_arr_copy.copy()
    indices_0 = np.argwhere(rotate90_arr1 < 255) # the indices of 0s
    rotate90_arr_copy[0][indices_0[:,0],indices_0[:,1]] = rotate90_arr[0][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[1][indices_0[:,0],indices_0[:,1]] = rotate90_arr[1][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[2][indices_0[:,0],indices_0[:,1]] = rotate90_arr[2][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[3][indices_0[:,0],indices_0[:,1]] = rotate90_arr[3][indices_0[:,0],indices_0[:,1]]
    
    # if len(rotate90_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    return rotate90_arr_copy

def rotate180(road_array,img_arr_copy):
    rotate90_arr1 = np.rot90(road_array[0],2)
    rotate90_arr2 = np.rot90(road_array[1],2)
    rotate90_arr3 = np.rot90(road_array[2],2)
    rotate90_arr4 = np.rot90(road_array[3],2)
    rotate90_arr = []
    rotate90_arr.append(rotate90_arr1)
    rotate90_arr.append(rotate90_arr2)
    rotate90_arr.append(rotate90_arr3)
    rotate90_arr.append(rotate90_arr4)
    rotate90_arr = np.array(rotate90_arr)
    
    # if len(rotate90_arr.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    rotate90_arr_copy = img_arr_copy.copy()
    indices_0 = np.argwhere(rotate90_arr1 < 255) # the indices of 0s
    rotate90_arr_copy[0][indices_0[:,0],indices_0[:,1]] = rotate90_arr[0][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[1][indices_0[:,0],indices_0[:,1]] = rotate90_arr[1][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[2][indices_0[:,0],indices_0[:,1]] = rotate90_arr[2][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[3][indices_0[:,0],indices_0[:,1]] = rotate90_arr[3][indices_0[:,0],indices_0[:,1]]
    
    # if len(rotate90_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    return rotate90_arr_copy

def rotate270(road_array,img_arr_copy):
    rotate90_arr1 = np.rot90(road_array[0],3)
    rotate90_arr2 = np.rot90(road_array[1],3)
    rotate90_arr3 = np.rot90(road_array[2],3)
    rotate90_arr4 = np.rot90(road_array[3],3)
    rotate90_arr = []
    rotate90_arr.append(rotate90_arr1)
    rotate90_arr.append(rotate90_arr2)
    rotate90_arr.append(rotate90_arr3)
    rotate90_arr.append(rotate90_arr4)
    rotate90_arr = np.array(rotate90_arr)
    
    # if len(rotate90_arr.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    rotate90_arr_copy = img_arr_copy.copy()
    indices_0 = np.argwhere(rotate90_arr1 < 255) # the indices of 0s
    rotate90_arr_copy[0][indices_0[:,0],indices_0[:,1]] = rotate90_arr[0][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[1][indices_0[:,0],indices_0[:,1]] = rotate90_arr[1][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[2][indices_0[:,0],indices_0[:,1]] = rotate90_arr[2][indices_0[:,0],indices_0[:,1]]
    rotate90_arr_copy[3][indices_0[:,0],indices_0[:,1]] = rotate90_arr[3][indices_0[:,0],indices_0[:,1]]
    
    # if len(rotate90_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(rotate90_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(rotate90_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    return rotate90_arr_copy

def flip_vertical(road_array,img_arr_copy):
    flip_arr1 = np.flip(road_array[0],0)
    flip_arr2 = np.flip(road_array[1],0)
    flip_arr3 = np.flip(road_array[2],0)
    flip_arr4 = np.flip(road_array[3],0)
    flip_arr = []
    flip_arr.append(flip_arr1)
    flip_arr.append(flip_arr2)
    flip_arr.append(flip_arr3)
    flip_arr.append(flip_arr4)
    flip_arr = np.array(flip_arr)
    
    # if len(flip_arr.shape) == 2:
    #     img_arr_plt = np.expand_dims(flip_arr, -1)
    # else:
    #     img_arr_plt = np.moveaxis(flip_arr, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    flip_arr_copy = img_arr_copy.copy()
    indices_0 = np.argwhere(flip_arr1 < 255) # the indices of 0s
    flip_arr_copy[0][indices_0[:,0],indices_0[:,1]] = flip_arr[0][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[1][indices_0[:,0],indices_0[:,1]] = flip_arr[1][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[2][indices_0[:,0],indices_0[:,1]] = flip_arr[2][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[3][indices_0[:,0],indices_0[:,1]] = flip_arr[3][indices_0[:,0],indices_0[:,1]]
    
    # if len(flip_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(flip_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(flip_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    return flip_arr_copy

def flip_horizontal(road_array,img_arr_copy):
    flip_arr1 = np.flip(road_array[0],1)
    flip_arr2 = np.flip(road_array[1],1)
    flip_arr3 = np.flip(road_array[2],1)
    flip_arr4 = np.flip(road_array[3],1)
    flip_arr = []
    flip_arr.append(flip_arr1)
    flip_arr.append(flip_arr2)
    flip_arr.append(flip_arr3)
    flip_arr.append(flip_arr4)
    flip_arr = np.array(flip_arr)
    
    # if len(flip_arr.shape) == 2:
    #     img_arr_plt = np.expand_dims(flip_arr, -1)
    # else:
    #     img_arr_plt = np.moveaxis(flip_arr, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    flip_arr_copy = img_arr_copy.copy()
    indices_0 = np.argwhere(flip_arr1 < 255) # the indices of 0s
    flip_arr_copy[0][indices_0[:,0],indices_0[:,1]] = flip_arr[0][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[1][indices_0[:,0],indices_0[:,1]] = flip_arr[1][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[2][indices_0[:,0],indices_0[:,1]] = flip_arr[2][indices_0[:,0],indices_0[:,1]]
    flip_arr_copy[3][indices_0[:,0],indices_0[:,1]] = flip_arr[3][indices_0[:,0],indices_0[:,1]]
    
    # if len(flip_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(flip_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(flip_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    return flip_arr_copy

###The initialization of data augmentation function
def data_aug_init(source_arr,topS,leftS,topT,leftT):
    # tif_file = r"data_augment_test.tif"
    # shp_file = r"road_buffer_7_data_augment.shp"
    buffer_tif = r"D:\Catherine\doctoral studies\road extraction\test\20220106\road_buffer.tif"
    
    input_patch = source_arr.copy()
    # img_arr   = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    
    ###To visualze the subimage
    # if len(input_patch.shape) == 2:
    #     input_plt = np.expand_dims(input_patch, -1)
    # else:
    #     input_plt = np.moveaxis(input_patch, 0, -1)
    # plt.imshow(input_plt)
    # plt.show()
    
    num_rows, num_cols = input_patch.shape[1],input_patch.shape[2]

    minx_i = leftS
    maxy_i = topS
    
    ### get the extent of road buffer image
    ds1 = gdal.Open(buffer_tif)
    buffer_arr   = np.array(ds1.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    buffer_arr = np.where(buffer_arr > 0.5, 0, 1)
    num_rows1, num_cols1 = buffer_arr.shape
    gt_buffer = ds1.GetGeoTransform()
    ext_buffer = GetExtent(gt_buffer,num_cols1,num_rows1)
    minx_b = ext_buffer[0][0]
    maxx_b = ext_buffer[2][0]
    miny_b = ext_buffer[1][1]
    maxy_b = ext_buffer[0][1]
    
    ###calculate the offset of top left point
    step_x = int(abs(minx_i - minx_b)/1.25)
    step_y = int(abs(maxy_i - maxy_b)/1.25)
    
    ### extract the corresponding area from the road buffer image
    extracted_buffer = buffer_arr[step_y:step_y+128,step_x:step_x+128]
    ###To visualize the road buffer image
    # plt.imshow(extracted_buffer)
    # plt.show()

    ### obtain the row and column indices of buffer pixels
    indices_0 = np.argwhere(extracted_buffer==0) # the indices of 1s
    input_patch_copy = input_patch.copy()
    # road_array[indices_1[:,0],indices_1[:,1],:] = input_patch[indices_1[:,0],indices_1[:,1]]
    input_patch_copy[0][indices_0[:,0],indices_0[:,1]] = 255
    input_patch_copy[1][indices_0[:,0],indices_0[:,1]] = 255
    input_patch_copy[2][indices_0[:,0],indices_0[:,1]] = 255
    input_patch_copy[3][indices_0[:,0],indices_0[:,1]] = 255

    ###To visualize the roads from subimage based on road buffer image
    # if len(input_patch_copy.shape) == 2:
    #     road_array_plt = np.expand_dims(input_patch_copy, -1)
    # else:
    #     road_array_plt = np.moveaxis(input_patch_copy, 0, -1)
    # plt.imshow(road_array_plt)
    # plt.show()
    
    # num_rows, num_cols = img_arr.shape
    # gt_img = ds.GetGeoTransform()
    
    ####fill in the original road area with background color
    indices_1 = np.argwhere(extracted_buffer==1)
    pixel_bgd1 = np.random.randint(low=230, high=245, size=len(indices_1))
    pixel_bgd2 = np.random.randint(low=230, high=240, size=len(indices_1))
    pixel_bgd3 = np.random.randint(low=220, high=230, size=len(indices_1))
    
    # pixel_bgd = 255
    img_arr_copy = input_patch.copy()
    img_arr_copy[0][indices_1[:,0],indices_1[:,1]] = pixel_bgd1
    img_arr_copy[1][indices_1[:,0],indices_1[:,1]] = pixel_bgd2
    img_arr_copy[2][indices_1[:,0],indices_1[:,1]] = pixel_bgd3
    img_arr_copy[3][indices_1[:,0],indices_1[:,1]] = 255
    
    ###visualize the non-road area
    # if len(img_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(img_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(img_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    return input_patch_copy, img_arr_copy
    
####conduct data autmentation operations randomly by generating random number
def data_aug_random(input_patch_copy, img_arr_copy,buffer,normalize=True):
    # plt.imshow(buffer)
    # plt.show()
    
    flag = random.randint(0,4)
    if flag == 0:
        arr_aug_source = rotate90(input_patch_copy,img_arr_copy)
        arr_aug_target = np.rot90(buffer,1)
        # plt.imshow(arr_aug_target)
        # plt.show()
    elif flag == 1:
        arr_aug_source = rotate180(input_patch_copy,img_arr_copy) 
        arr_aug_target = np.rot90(buffer,2)
        # plt.imshow(arr_aug_target)
        # plt.show()
    elif flag == 2:
        arr_aug_source = rotate270(input_patch_copy,img_arr_copy)
        arr_aug_target = np.rot90(buffer,3)
        # plt.imshow(arr_aug_target)
        # plt.show()
    elif flag == 3:
        arr_aug_source = flip_vertical(input_patch_copy,img_arr_copy)
        arr_aug_target = np.flip(buffer,0)
        # plt.imshow(arr_aug_target)
        # plt.show()
    else:
        arr_aug_source = flip_horizontal(input_patch_copy,img_arr_copy)
        arr_aug_target = np.flip(buffer,1)
        # plt.imshow(arr_aug_target)
        # plt.show()
        
    if len(arr_aug_source.shape) == 2:
        arr_aug_source = np.expand_dims(arr_aug_source, -1)
    else:
        arr_aug_source = np.moveaxis(arr_aug_source, 0, -1)
    if normalize:
        arr_aug_source = arr_aug_source / 255
    
    if len(arr_aug_target.shape) == 2:
        arr_aug_target = np.expand_dims(arr_aug_target, -1)
    else:
        arr_aug_target = np.moveaxis(arr_aug_target, 0, -1)
    if normalize:
        arr_aug_target = arr_aug_target / 255
    return arr_aug_source,arr_aug_target
            
            
        
    

if __name__ == '__main__':
    # tif_file = r"data_augment_test.tif"
    # # shp_file = r"road_buffer_7_data_augment.shp"
    # buffer_tif = r"road_buffer_7.tif"
    
    # ### get the extent of subimage
    # ds = gdal.Open(tif_file)
    # input_patch = ds.ReadAsArray()
    # img_arr   = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint8)

    # if len(input_patch.shape) == 2:
    #     input_plt = np.expand_dims(input_patch, -1)
    # else:
    #     input_plt = np.moveaxis(input_patch, 0, -1)
    # plt.imshow(input_plt)
    # plt.show()
    # num_rows, num_cols = img_arr.shape
    # gt_img = ds.GetGeoTransform()
    # ext_img = GetExtent(gt_img,num_cols,num_rows)
    # minx_i = ext_img[0][0]
    # maxx_i = ext_img[2][0]
    # miny_i = ext_img[1][1]
    # maxy_i = ext_img[0][1]
    
    # ### get the extent of road buffer image
    # ds1 = gdal.Open(buffer_tif)
    # buffer_arr   = np.array(ds1.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    # buffer_arr = np.where(buffer_arr > 0.5, 0, 1)
    # num_rows1, num_cols1 = buffer_arr.shape
    # gt_buffer = ds1.GetGeoTransform()
    # ext_buffer = GetExtent(gt_buffer,num_cols1,num_rows1)
    # minx_b = ext_buffer[0][0]
    # maxx_b = ext_buffer[2][0]
    # miny_b = ext_buffer[1][1]
    # maxy_b = ext_buffer[0][1]
    
    # ###calculate the offset of top left point
    # step_x = int(abs(minx_i - minx_b)/1.25)
    # step_y = int(abs(maxy_i - maxy_b)/1.25)
    
    # ### extract the corresponding area from the road buffer image
    # # extracted_buffer = buffer_arr[step_x:step_x+128,step_y:step_y+128]
    # extracted_buffer = buffer_arr[step_y:step_y+128,step_x:step_x+128]
    # plt.imshow(extracted_buffer)
    # plt.show()

    # ### obtain the row and column indices of buffer pixels
    # indices_0 = np.argwhere(extracted_buffer==0) # the indices of 1s
    # input_patch_copy = input_patch.copy()
    # # road_array[indices_1[:,0],indices_1[:,1],:] = input_patch[indices_1[:,0],indices_1[:,1]]
    # input_patch_copy[0][indices_0[:,0],indices_0[:,1]] = 255
    # input_patch_copy[1][indices_0[:,0],indices_0[:,1]] = 255
    # input_patch_copy[2][indices_0[:,0],indices_0[:,1]] = 255
    # input_patch_copy[3][indices_0[:,0],indices_0[:,1]] = 255

    # if len(input_patch_copy.shape) == 2:
    #     road_array_plt = np.expand_dims(input_patch_copy, -1)
    # else:
    #     road_array_plt = np.moveaxis(input_patch_copy, 0, -1)
    # plt.imshow(road_array_plt)
    # plt.show()
    
    # num_rows, num_cols = img_arr.shape
    # gt_img = ds.GetGeoTransform()
    
    # ####fill in the road area with background color
    # indices_1 = np.argwhere(extracted_buffer==1)
    # pixel_bgd1 = np.random.randint(low=230, high=245, size=len(indices_1))
    # pixel_bgd2 = np.random.randint(low=230, high=240, size=len(indices_1))
    # pixel_bgd3 = np.random.randint(low=220, high=230, size=len(indices_1))
    
    # # pixel_bgd = 255
    # img_arr_copy = input_patch.copy()
    # img_arr_copy[0][indices_1[:,0],indices_1[:,1]] = pixel_bgd1
    # img_arr_copy[1][indices_1[:,0],indices_1[:,1]] = pixel_bgd2
    # img_arr_copy[2][indices_1[:,0],indices_1[:,1]] = pixel_bgd3
    # img_arr_copy[3][indices_1[:,0],indices_1[:,1]] = 255
    
    # if len(img_arr_copy.shape) == 2:
    #     img_arr_plt = np.expand_dims(img_arr_copy, -1)
    # else:
    #     img_arr_plt = np.moveaxis(img_arr_copy, 0, -1)
    # plt.imshow(img_arr_plt)
    # plt.show()
    
    input_patch_copy,img_arr_copy =  data_aug_init()
    ###rotate raods 90 degree 
    arr_rptate90 = rotate90(input_patch_copy,img_arr_copy)
    arr_rptate180 = rotate180(input_patch_copy,img_arr_copy)
    arr_rptate270 = rotate270(input_patch_copy,img_arr_copy)
    
    arr_flip_h = flip_vertical(input_patch_copy,img_arr_copy)
    arr_flip_v = flip_horizontal(input_patch_copy,img_arr_copy)

    


