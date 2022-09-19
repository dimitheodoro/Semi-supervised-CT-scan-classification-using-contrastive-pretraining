import numpy as np
from scipy import ndimage
!pip install pydicom
import  pydicom as dcm
import matplotlib.pyplot as plt

# Function to take care of teh translation and windowing. 
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
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




def read_dicom_images(path):
    scans = [dcm.read_file(os.path.join(path,slice)) for slice in os.listdir(path)]
    slices = np.array([dcm.read_file(os.path.join(path,slice)).pixel_array for slice in os.listdir(path)])
    slices =np.transpose(slices,(1,2,0))
    window_center , window_width, intercept, slope = get_windowing(scans[0])  
    return  window_image(slices,window_center , window_width, intercept, slope )


def resize_volume(img):
    print(img.shape)
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    print(img.shape)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_dicom_images(path)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


new_volume =process_scan('lung-ct.volume-3d')
