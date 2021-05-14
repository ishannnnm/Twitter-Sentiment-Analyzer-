import tweepy
from textblob import TextBlob

consumer_key = 'sUtcYouJYEhNG7FuSSPn7fzZF'
consumer_secret = 'WT82pwnPpZXmGbaycXCKsCVzDEG17LMTUziabU2ix9juvkY6mf'

access_token = '526402213-Hmhv8Tu0QWGY27A5mzEDviCePEn90mXfihNPou7B'
access_token_secret = '6AXy26YVRO69pQxfpN55gkajv4IJv62L4EZ3yvjM5btLy'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('modi')

for tweet in public_tweets :
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

DATA_DIR = '/kaggle/input/siim-acr-pneumothorax-segmentation-data/pneumothorax'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'

ls {DATA_DIR}
# !pip install 'keras==2.1.6' --force-reinstall
STAGE_DIR = '/tmp/Mask_RCNN'
!git clone https://www.github.com/matterport/Mask_RCNN.git {STAGE_DIR}
os.chdir(STAGE_DIR)
#!python setup.py -q install
!rm .git samples images assets -rf
!pwd; ls

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-train')
test_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-test')

# count files
!ls -m {train_dicom_dir} | wc
!ls -m {test_dicom_dir} | wc
# get model with best validation score: https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155/
WEIGHTS_PATH = "mask_rcnn_pneumonia.h5"
!cp /kaggle/input/mask-rcnn*/pneumonia*/*0013.h5 {WEIGHTS_PATH}
!du -sh *.h5

# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal

IMAGE_DIM = 512

class DetectorConfig(Config):    
    # Give the configuration a recognizable name  
    NAME = 'Pneumothorax'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 11
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background and pneumothorax classes
    
    IMAGE_MIN_DIM = IMAGE_DIM
    IMAGE_MAX_DIM = IMAGE_DIM
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 12
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.90
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 20 if debug else 320
    VALIDATION_STEPS = 10 if debug else 100
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 10.0,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 2.4
    }

config = DetectorConfig()
config.display()
