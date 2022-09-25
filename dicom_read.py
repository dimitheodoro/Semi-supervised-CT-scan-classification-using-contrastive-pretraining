import numpy as np
from scipy import ndimage
!pip install pydicom
import  pydicom as dcm
import matplotlib.pyplot as plt

############################  WITH  SORTED SLICCES ######################################## 


path ='lung-ct.volume-3d'

def sort_dicoms(path):
    acq_times = ([dcm.read_file(os.path.join(path,slice))[('0008','0032')].value for slice in (os.listdir((path)))])
    scans_times = dict(zip(os.listdir(path),acq_times))
    scans_times = {k: v for k, v in sorted(scans_times.items(), key=lambda item: item[1])}
    sorted_dcm = list(scans_times.keys())
    return sorted_dcm

sorted_dcm = sort_dicoms(path)


# Function to take care of teh translation and windowing. 
def window_image(img, window_center,window_width, intercept, slope, rescale=False):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def read_dicom_images(path,sorted_dcm):
    scans = [dcm.read_file(os.path.join(path,slice)) for slice in sorted_dcm]
    slices =  np.array([dcm.read_file(os.path.join(path,slice)).pixel_array for slice in sorted_dcm])
    # slices =np.transpose(slices,(1,2,0))
    window_center , window_width, intercept, slope = get_windowing(scans[0])  
    return  window_image(slices,window_center , window_width, intercept, slope )


def process_scan(path,sorted_dcm):
    """Read and resize volume"""
    # Read scan
    volume = read_dicom_images(path,sorted_dcm)
    # Resize width, height and depth
    return volume

slices =process_scan('lung-ct.volume-3d',sorted_dcm)


print(slices.shape)
########################################## resampling ###############################################
def get_spacing(path):

    pixel_spacing = [dcm.read_file(os.path.join(path,slice)).PixelSpacing for slice in (os.listdir((path)))][:1]
    slice_thickness = [dcm.read_file(os.path.join(path,slice)).SliceThickness for slice in (os.listdir((path)))][:1]


    # one_axis_spacing = np.array(np.float32(slice_thickness)+pixel_spacing)[0][0]
    # pixel_spacing = [dcm.read_file(os.path.join(path,slice))[('0028','0030')].value for slice in (os.listdir((path)))][:1]
    # slice_thickness = [dcm.read_file(os.path.join(path,slice))[('0018','0050')].value for slice in (os.listdir((path)))][:1]
    # # one_axis_spacing = np.array(np.float32(slice_thickness)+pixel_spacing)[0][0]
    

    return  pixel_spacing,slice_thickness
                          
pixel_spacing,slice_thickness = get_spacing(path)
# print(pixel_spacing,slice_thickness)

def resample(image,pixel_spacing, slice_thickness , new_spacing=[1,1,1]):

    spacing = np.array([slice_thickness[0],pixel_spacing[0][0],pixel_spacing[0][1]])
    resize_factor = spacing / np.array(new_spacing)  
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor    
    print("new_spacing:",new_spacing)
    image = zoom(image, real_resize_factor, mode='nearest')
  
    
    return image

resampled_slices=resample (slices,pixel_spacing, slice_thickness)

resampled_slices.shape


