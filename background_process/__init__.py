
  
__version__ = '1.0'
__description__ = 'A module to process background remove or change'
__dependencies__ = ['requests']
__author__ = 'HaiND-VFAST'
__url__ = 'https://git.vfastsoft.com/VFAST/Background_removal.git'
__license__ = 'GPLv3'


from .processing import background_img, background_vid
try:
    from .object_segment import load_model
except:
    from object_segment import load_model

