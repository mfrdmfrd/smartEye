import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth,DBSCAN
from itertools import cycle
import os
#from osgeo import gdal,ogr,osr
import time
import sqlite3
import io
from math import cos, sin, asin, sqrt, radians
import pickle
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans as KMeans
import logging
import sys
from sklearn.decomposition import RandomizedPCA
import copy
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import encoding


global data_root,Trainingdir_in,Testingdir_in,log_file,out_dir,db_filepath, LOCALIZER_WINDOW,LOCALIZER_MEANSHIFT,LOCALIZER_DBSCAN,localizer,gray

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def initialize_parameteres(data_root_='.\\Data\\',dbname="train.db"):
    global data_root,Trainingdir_in,Testingdir_in,log_file,out_dir,db_filepath
    global LOCALIZER_WINDOW,LOCALIZER_MEANSHIFT,LOCALIZER_DBSCAN,localizer,gray
    global epsilon,min_pts_per_cluster,accuracy,retrain,graph
    global window_overlapping_factor,window_size_factor
    global reduced,bow_encoding

# Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    data_root = data_root_
   
    out_dir=os.path.join(data_root,"ops_out")
    os.makedirs(out_dir, exist_ok=True)

    Trainingdir_in = os.path.join(data_root, 'Training')
    Testingdir_in = os.path.join(data_root , 'Testing')
    #os.makedirs(Testingdir_in, exist_ok=True)


    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    db_filepath=os.path.join(data_root,dbname)
    conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

    LOCALIZER_WINDOW=0
    LOCALIZER_MEANSHIFT=1
    LOCALIZER_DBSCAN=2

    try:
        conn.execute('''SELECT * FROM category;''')
        print ("Database Opened successfully")
    except:
        f = open('db.sql', 'r')   
        
        conn.executescript(f.read())
        print ("Database Created successfully")
        conn.commit()


def initialize_parameteres_han(data_root_='E:\\Master\\Data\\',dbName='train.db'):
    global data_root,Trainingdir_in,Testingdir_in,log_file,out_dir,db_filepath
    data_root = data_root_
   
    out_dir=os.path.join(data_root,"ops_out")
    os.makedirs(out_dir, exist_ok=True)

    Trainingdir_in = os.path.join(data_root, 'TrainSet')
    Testingdir_in = os.path.join(data_root , 'TestSet')
    #os.makedirs(Testingdir_in, exist_ok=True)


    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    db_filepath=os.path.join(data_root,dbName)
    conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
    

    try:
        conn.execute('''SELECT * FROM category;''')
        print ("Database Opened successfully")
    except:
        f = open('db.sql', 'r')   
        
        conn.executescript(f.read())
        print ("Database Created successfully")
        conn.commit()


def detect_compute_features(img,detector,descriptor):

    kp=detector.detect(img,None)
    des = descriptor.compute(img,kp,None)
    
    return kp,des



#### Filter Weak cluststers
def filter_features_clusters3(cluster_centers,labels,score):
    log('Filter weak objects Using SVM Classifier...')

    cluster_center_filtered=[]
    
    for k in range(len(cluster_centers)):
        if  score[k] > 0:
            cluster_center_filtered.append(1)
        else:
            cluster_center_filtered.append(0)
    return cluster_center_filtered


#### Filter Weak cluststersve
def filter_features_clusters(cluster_centers,labels,MIN_MATCH_COUNT=3):
    log('Filter weak objects...')

    cluster_center_filtered=[]
    
    for k in range(len(cluster_centers)):
        my_members = labels == k
        members_count=np.sum(my_members)
        if members_count >= MIN_MATCH_COUNT:
            #Delete from labels and src_pts the filtered classes
            #cluster_center_filtered.append(cluster_centers[k])
            cluster_center_filtered.append(1)
        else:
            cluster_center_filtered.append(0)
    return cluster_center_filtered

def filter_features_clusters2(cluster_centers,labels,kps,bandwidth):
    log('Filter weak objects...')

    cluster_center_filtered=[]
    
    for k in range(len(cluster_centers)):
        for i in range(len(labels)):
            if labels[i]==k:
                my_members.append([kps[0][i].x,kps[0][i].y])

        hull = cv2.convexHull(my_members)
        M = cv2.moments(hull)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull,True)
        if area >= bandwidth**2:
            #Delete from labels and src_pts the filtered classes
            cluster_center_filtered.append(cluster_centers[k])
          
    return cluster_center_filtered
    
#def plot_clusters_features(img,cluster_centers,labels,src_pts,savefig=0,img_name='image'):
#    plt.imshow(img)
#    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#    n_clusters_=len(cluster_centers)
#    for k, col in zip(range(n_clusters_), colors):
#        my_members = labels == k
#        cluster_center = cluster_centers[k]
#        plt.plot(src_pts[my_members, 0], src_pts[my_members, 1], col + '.')
#    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    if savefig:
#        plt.savefig('clusters_p_'+img_name+'.png')
#        log('clustering result image saved')
#    plt.show()


# In[9]:

#def plot_features_cluster_centers(img,cluster_centers,savefig=0,img_name='image'):
#    plt.imshow(img)
#    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#    n_clusters_=len(cluster_centers)
#    for k, col in zip(range(n_clusters_), colors):
#        cluster_center = cluster_centers[k]
#        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='none',
#                 markeredgecolor='r', markersize=20)
#    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    if savefig:
#        plt.savefig(os.path.join(p.Testingdir_out,'results_'+img_name+str(time.time())+'.png'))
#        log('clustering result image saved')

#    if savefig > 1:
#        plt.show()

def plot_features_and_cluster_centers(img,cluster_centers,cc_f,labels,src_pts,savefig=0,img_name='',title=''):
    
    if len(img.shape)==3 :
        b,g,r = cv2.split(img)       # get b,g,r    
        plt.imshow(cv2.merge([r,g,b]))
    else:
        plt.imshow(img)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    n_clusters_ = len(cluster_centers)
    count=0
    for k, col in zip(range(n_clusters_), colors):
        if cc_f is None or cc_f==[] or cc_f[k] > 0:
            cluster_center = cluster_centers[k]
            my_members = labels == k
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='none',
                        markeredgecolor='r', markersize=20)
            if labels!=[]:
                plt.plot(src_pts[my_members, 0], src_pts[my_members, 1], col + '.')
            count+=1
    plt.axis('off')
    #plt.title(title)

    plt.title('Number of clusters: %d' % count)
    if savefig:
        plt.savefig(img_name,bbox_inches='tight')
        #log('clustering result image saved')

    if savefig > 1:
        plt.show()
    plt.close()

def plot_features2(img,key_pts,savefig=1,imagepath='unknown_path.png',title='Plot Features 2'):
    
    if savefig:
        if len(img.shape)==3 :
            b,g,r = cv2.split(img)       # get b,g,r    
            img2=cv2.merge([r,g,b])
        else:
            img2=img
            
        src_pts=[kp.pt for kp in key_pts] #get coordinates from ketpoint class 
        if src_pts!=[]:
            x, y = zip(*src_pts)
        else:
            x,y=[],[]
    
        plt.imshow(img2)
        plt.title(title)
        plt.axis('off')
        if len(src_pts) > 0:
            plt.plot(x,y, 'r' + '.')
        plt.savefig(imagepath,bbox_inches='tight')
        if savefig>1:
            plt.show()
        plt.close()


    #sizes = np.shape(img2)
    #height = float(sizes[0])
    #width = float(sizes[1])
     
    #fig = plt.figure()
    #fig.set_size_inches(width/height, 1, forward=False)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
 
    #ax.imshow(img2)
    
    #plt.savefig(imagePath)
    #plt.show() 
    #plt.close()


def plot_features(img,src_pts,savefig=1,imagePath=''):
    if len(img.shape)==3 :
        b,g,r = cv2.split(img)       # get b,g,r    
        plt.imshow(cv2.merge([r,g,b]))
    else:
        plt.imshow(img)

    if len(src_pts) > 0:
        plt.plot(src_pts[:,0],src_pts[:,1], 'r' + '.')
    plt.title('Good Features')
    if savefig:
        plt.savefig(imagePath,bbox_inches='tight')
    plt.close()

def load_truth_table(img_file_name):
    TT=np.loadtxt(img_file_name,delimiter=',', ndmin=2)
    #log('Truth table loaded.')

    return TT

def load_truth_table2(gtfile):
    TT=[]
    f = open(gtfile, "r")
    for line in f.readlines():
        if line != '\n' and line !='':
            l=line.split(',')
            del l[-1]
            for i in range (4):
                l[i]=int(l[i].strip('()'))
            x=l[0]+(l[2]-l[0])/2
            y=l[1]+(l[3]-l[1])/2         
            TT.append([x,y])
    return TT

def find_nearest_position(value,array):
    #print ('Array-Value',array-value)
    dist_a = np.array([np.linalg.norm([x,y]) for (x,y) in array-value])
    #print (dist_a)
    idx = dist_a.argmin()
    distance=calc_distance(value[0],value[1],array[idx][0],array[idx][1])
    return idx,distance

def find_nearest_position2(value,array):
    dist_a = np.array([np.linalg.norm([x,y]) for (x,y) in array-value])
    
    #print (dist_a)
    idx = dist_a.argmin()
    distance=calc_distance2(value[0],value[1],array[idx][0],array[idx][1])
    return idx,distance

def find_nearest_position3(value,array,matched):
    dist_a = np.array([np.linalg.norm([x,y]) for (x,y) in array-value])
    idx = dist_a.argmin()
    
    while matched[idx]==1 and dist_a[idx]<1000000:
        dist_a[idx]=1000000
        idx = dist_a.argmin()

    return idx,dist_a[idx]
#def pickle_keypoints(descriptors):
 
#    temp_array = []
#    if descriptors[1] != None:
#        for point,des in zip(descriptors[0],descriptors[1]):
#            temp = (point.pt, point.size, point.angle, point.response, point.octave,
#            point.class_id, des) 
#            temp_array.append(temp)
#    return temp_array

#def unpickle_keypoints(array):
#    keypoints = []
#    descriptors = []
#    for point in array:
#        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
#        temp_descriptor = point[6]
#        keypoints.append(temp_feature)
#        descriptors.append(temp_descriptor)
#    return keypoints, np.array(descriptors)

#def store_kps():
#    #Store keypoint features
#    temp_array = []
#    temp = pickle_keypoints(kp1, desc1)
#    temp_array.append(temp)
#    temp = pickle_keypoints(kp2, desc2)
#    temp_array.append(temp)
#    pickle.dump(temp_array, open("keypoints_database.p", "wb"))

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

# The following method translates given latitude/longitude pairs into pixel locations on a given GEOTIF
# INPUTS: geotifAddr - The file location of the GEOTIF
#      latLonPairs - The decimal lat/lon pairings to be translated in the form [[lat1,lon1],[lat2,lon2]]
# OUTPUT: The pixel translation of the lat/lon pairings in the form [[x1,y1],[x2,y2]]
# NOTE:   This method does not take into account pixel size and assumes a high enough 
#	  image resolution for pixel size to be insignificant
def latLonToPixel(geotifAddr, latLonPairs):
	# Load the image dataset
	ds = gdal.Open(geotifAddr)
	# Get a geo-transform of the dataset
	gt = ds.GetGeoTransform()
	# Create a spatial reference object for the dataset
	srs = osr.SpatialReference()
	srs.ImportFromWkt(ds.GetProjection())
	# Set up the coordinate transformation object
	srsLatLong = srs.CloneGeogCS()
	ct = osr.CoordinateTransformation(srsLatLong,srs)
	# Go through all the point pairs and translate them to latitude/longitude pairings
	pixelPairs = []
	for point in latLonPairs:
		# Change the point locations into the GeoTransform space
		(point[1],point[0],holder) = ct.TransformPoint(point[1],point[0])
		# Translate the x and y coordinates into pixel values
		x = (point[1]-gt[0])/gt[1]
		y = (point[0]-gt[3])/gt[5]
		# Add the point to our return array
		pixelPairs.append([int(x),int(y)])
	return pixelPairs
# The following method translates given pixel locations into latitude/longitude locations on a given GEOTIF
# INPUTS: geotifAddr - The file location of the GEOTIF
#      pixelPairs - The pixel pairings to be translated in the form [[x1,y1],[x2,y2]]
# OUTPUT: The lat/lon translation of the pixel pairings in the form [[lat1,lon1],[lat2,lon2]]
# NOTE:   This method does not take into account pixel size and assumes a high enough 
#	  image resolution for pixel size to be insignificant
def pixelToLatLon_(geotifAddr,pixelPairs,mask=None):
    # Load the image dataset
    ds = gdal.Open(geotifAddr)
    # Get a geo-transform of the dataset
    gt = ds.GetGeoTransform()
    # Create a spatial reference object for the dataset
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    # Set up the coordinate transformation object
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs,srsLatLong)
    # Go through all the point pairs and translate them to pixel pairings
    latLonPairs = []
    for i in range(len(pixelPairs)):
        if mask is None or mask[i]==1:
            point = pixelPairs[i]
            if gt[0] != 0:
                # Translate the pixel pairs into untranslated points
                ulon = point[0]*gt[1]+gt[0]
                ulat = point[1]*gt[5]+gt[3]
	            # Transform the points to the space
                (lon,lat,holder) = ct.TransformPoint(ulon,ulat)
	            # Add the point to our return array
                latLonPairs.append([lat,lon])
            else:
                latLonPairs.append(point)
    return latLonPairs

def pixelToLatLon(geotifAddr,pixelPairs,mask=None):
    
    latLonPairs = []
    for i in range(len(pixelPairs)):
        if mask is None or mask[i]==1:
            point = pixelPairs[i]
            latLonPairs.append(point)
    return latLonPairs

def log(message,file_=None,print_=1):
    if print_>0:
        print (message)
    if file_:
        file_.write(message + '\n')


def calc_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    meters = 6371 * c *1000
    return meters

def calc_distance2(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    #lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    dist=sqrt(pow(dlon,2) + pow(dlat,2))
    
    return dist
#def calc_backprojection(roi,target):
#    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

#    # calculating object histogram
#    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

#    # normalize histogram and apply backprojection
#    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
#    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 
#    # Now convolute with circular disc
#    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#    cv2.filter2D(dst,-1,disc,dst)
 
#    # threshold and binary AND
#    ret,thresh = cv2.threshold(dst,50,255,0)
#    thresh = cv2.merge((thresh,thresh,thresh))
#    res = cv2.bitwise_and(target,thresh)


#def compareHistogrm(images,targets):#images: training samples, targets: detected targets in the testing image
#    hist = cv2.calcHist(images, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#    hist = cv2.normalize(hist,None).flatten()
#    plt.plot(hist)
#    plt.xlim([0,256])
#    plt.show()
    
#    result=[]

#    for t in targets:
#        hist_t = cv2.calcHist(t, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#        hist_t = cv2.normalize(hist_t,None).flatten()
#        plt.plot(hist_t)
#        plt.xlim([0,256])
#        plt.show()
#        d = cv2.compareHist(hist, hist_t, cv2.HISTCMP_CHISQR)
#        result.append(d)

#    return result

#def getPixelSizeMeters(img_path):
#    """Calculate the real width and height of the sample"""
#    gimg=gdal.Open(img_path)
#    gt=gimg.GetGeoTransform()
#    src = osr.SpatialReference()
#    wkt=gimg.GetProjection()
#    src.ImportFromWkt(wkt)
#    unit=src.GetAttrValue("UNIT")
#    Xsize=gimg.RasterXSize
#    Ysize=gimg.RasterYSize
#    X=gt[0]             #ToDo : make sure X,Y not switched -Lat,long assoctiate
#    Y=gt[3]
        
#    if unit == 'degree':
#        dX=calc_distance(X,Y,X+gt[1],Y)
#        dY=calc_distance(X,Y,X,Y+gt[5])
#    else:
#        dX=abs(gt[1])
#        dY=abs(gt[5])
        
#    return dX,dY

#def filterByHistogram(category,cluster_centers,img,img_path):
#    #images: training samples, targets: detected targets in the testing image
#    images=[s.img for s in category.samples]
#    hist = cv2.calcHist(images, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#    hist = cv2.normalize(hist,None).flatten()
#    plt.plot(hist)
#    plt.xlim([0,256])
#    plt.show()

#    targets=[]
#    b_meters=category.bandwidth/2
#    pxSize=max(getPixelSizeMeters(img_path))
#    b=b_meters/pxSize       #bandwidth with pixels

#    for cc in cluster_centers:
#        y1=int(cc[1]-b)
#        if y1<0: y1=0
#        y2=int(cc[1]+b)
#        x1=int(cc[0]-b)
#        if x1<0: x1=0
#        x2=int(cc[0]+b)
#        target=img[y1:y2,x1:x2]
#        #plt.imshow(target)
#        #plt.show()      #todo comment
#        #targets.append(target)
#        hist_t = cv2.calcHist(target, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#        hist_t = cv2.normalize(hist_t,None).flatten()
#        #plt.plot(hist_t)
#        #plt.xlim([0,256])
#        #plt.show()
#        d = cv2.compareHist(hist, hist_t, cv2.HISTCMP_CHISQR)
#        if d > 0.5:
#            targets.append(cc)

#    return targets







class Sample:
    def __init__(self, file_name,file_path,aoi=[]):
        self.file_name = file_name
        
        self.file_path = os.path.join(file_path,file_name)
        self.hist=[]
        
        # img=
        # if (aoi==[]):
            # self.img=img
        # else:  
            # #h,w,d=img.shape
            # #img_mask=np.zeros((h,w), np.uint8)
            # #cv2.rectangle(img_mask,(aoi[0],aoi[1]),(aoi[2],aoi[3]),255,-1)
            # self.img=img[aoi[1]:aoi[3],aoi[0]:aoi[2]]
 

        self.aoi=aoi       
        #self.gimg=gdal.Open(self.file_path)       

        #self.__calc_dimensions()
        self.kps= []
 
    def get_img(self):
        img=cv2.imread(self.file_path,cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if (self.aoi==[]):
            return img
        else:  
            return img[self.aoi[1]:self.aoi[3],self.aoi[0]:self.aoi[2]]

    def set_category(self,c):
        self.category = c 

    def get_path(self):
        return os.path.join(cat_in_path, file)

    def calc_dimensions(self):
        """Calculate the real width and height of the sample"""
        gimg=self.get_img()
 		#gimg=gdal.Open(self.file_path) 
        #gt=gimg.GetGeoTransform()
        #src = osr.SpatialReference()
        #wkt=gimg.GetProjection()
        
        if gray:
            rows,cols=gimg.shape
        else:
            rows,cols,ch=gimg.shape

        gimg=None
        if (self.aoi==[]):
            self.width=col
            self.height=row
        else:
            self.width=self.aoi[2]-self.aoi[0]
            self.height=self.aoi[3]-self.aoi[1]
        #if (wkt !=''):
        #    src.ImportFromWkt(wkt)
        #    unit=src.GetAttrValue("UNIT")
        #    X=gt[0]
        #    Y=gt[3]
        
        #    if unit == 'degree':
        #        dX=calc_distance(X,Y,X+gt[1],Y)
        #        dY=calc_distance(X,Y,X,Y+gt[5])
        #    else:
        #        dX=abs(gt[1])
        #        dY=abs(gt[5])
        
        #    self.width=Xsize*dX
        #    self.height=Ysize*dY
        #else:
        # self.width=Xsize
        # self.height=Ysize
           

    def get_geo_dims(self):
        return self.width,self.height

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

class Algorithm:
    def __init__(self, det_name,des_name=None,k=1000,n=0,matcher="FLANN"):
        self.detector_name = det_name
        self.des_name = des_name

        self.get_id()
        
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        if matcher == 'FLANN':
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params) 
        else:
            self.matcher=cv2.BFMatcher()

        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()
        cur.execute("select id from algorithm where name = ?",(det_name,))
        exist=cur.fetchone()
        if exist:
            self.id=exist[0]
        else:
            conn.execute('insert into algorithm values(?)',(det_name,))
            conn.commit()
            cur.execute("select id from algorithm where name = ?",(det_name,))
            alg_id=cur.fetchone()[0]
        conn.close()

        for case in switch(det_name):
            if case('SIFT'):
                self.detector=cv2.xfeatures2d.SIFT_create()
                break
            if case('SURF'):
                self.detector=cv2.xfeatures2d.SURF_create()
                break
            if case(): # default, could also just omit condition or 'if True'
                self.detector=cv2.xfeatures2d.SIFT_create()

        for case in switch(des_name):
            if case('SIFT'):
                self.descriptor=cv2.xfeatures2d.SIFT_create()
                self.des_name='SIFT'
                break
            if case('SURF'):
                self.descriptor=cv2.xfeatures2d.SURF_create()
                self.des_name='SURF'
                break
            if case(None): # default, could also just omit condition or 'if True'
                self.descriptor=self.detector
                self.des_name=self.detector_name
        self.k=k
        #self.bowextractor = cv2.BOWImgDescriptorExtractor(self.descriptor,self.matcher)
        self.n=n
        if bow_encoding == 1:
            self.bowextractor = encoding.HardHistogramEncoder()
        elif bow_encoding == 2:
            self.bowextractor = encoding.SoftHistogramEncoder()
        elif bow_encoding == 3:
            self.bowextractor = encoding.LLCEncoder()
        elif bow_encoding == 0:
            self.bowextractor = cv2.BOWImgDescriptorExtractor(self.descriptor,self.matcher)

    def get_hist(self,img,kps,descs,vocab):
        if bow_encoding == 1:
            self.bowextractor = encoding.HardHistogramEncoder()
            hist=self.bowextractor.compute(descs,vocab)
        elif bow_encoding == 2:
            self.bowextractor = encoding.SoftHistogramEncoder()
            hist=self.bowextractor.compute(descs,vocab)
        elif bow_encoding == 3:
            self.bowextractor = encoding.LLCEncoder()
            hist=self.bowextractor.compute(descs,vocab)
        elif bow_encoding == 0:
            self.bowextractor = cv2.BOWImgDescriptorExtractor(self.descriptor,self.matcher)
            self.bowextractor.setVocabulary(vocab)
            hist=self.bowextractor.compute(img,kps,descs)


        return hist

    #def get_algorithm(self):
    #    sift=cv2.xfeatures2d.SIFT_create()
    #    surf=cv2.xfeatures2d.SURF_create()
    #    freak=cv2.xfeatures2d.FREAK_create()
    #    brisk=cv2.BRISK_create()
    #    brief=cv2.xfeatures2d.FREAK_create()
    #    #lucid=cv2.xfeatures2d.LUCID_create()
    #    star=cv2.xfeatures2d.StarDetector_create()
    #    orb=cv2.ORB_create()
    #    mser=cv2.MSER_create()
    #    sblob=cv2.SimpleBlobDetector_create()
    #    gftt=cv2.GFTTDetector_create()
    #    fast=cv2.FastFeatureDetector_create()

    #    #bow=cv2.BOWImgDescriptorExtractor()

    #    #kaze,Akaze --> C++ only


    #    #detectors=[Algorithm('SIFT',sift)]#,surf,star,orb,mser,gftt,fast]
    #    #descriptors=[sift]#,surf,freak,brief,sift,sift,orb]
    
    #    return Algorithm('SIFT',sift)
    def get_id(self):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        
        cur=conn.cursor()
            
        cur.execute("select id from algorithm where name = ?",(self.detector_name,))
        exist=cur.fetchone()
        if exist:
            alg_id=exist[0]
        else:
            conn.execute('insert into algorithm values(?)',(self.detector_name,))
            conn.commit()
            cur.execute("select id from algorithm where name = ?",(self.detector_name,))
            self.id=cur.fetchone()[0]

    def get_kps(self,img):          
           
        kp=self.detector.detect(img,None)
        des = self.descriptor.compute(img,kp)

        #if des[1] != None :
        #    training_data[0].extend(des[0])
        #    training_data[1].extend(des[1])
    
        return des#training_data 


class Category:
    """Object Category class"""

    #def __init__(self,name):
    #    self.name=name
    #    self.samples=[]
    #    self.load_samples()
    #    self.training_data= [[],[]]     #All kps of the all samples in the category
    #    self.results=[]                 #All detected objects for that category in all tiles
    #    self.bandwidth=0

    def __init__(self,id,name,bandwidth=0,SVM=None,PCA=None,train=1):
        self.id=id
        self.name=name
        self.samplesPos=[]
        self.samplesNeg=[]   
        self.training_data_pos=[]
        self.training_data_neg=[]
        self.training_data_all=[]
     
        self.cat_traindata_path=os.path.join(Trainingdir_in, self.name)
        self.cat_testdata_path=os.path.join(Testingdir_in, self.name)
        self.reduced_training_data_pos=[]
        self.reduced_training_data_neg=[]

        self.SVMTrainLabel=[]
        self.SVMTrainData=[]
        self.vocab=[]

        self.PCA=PCA
        self.SVM=SVM

        #if SVM is not None:
        #    self.SVM=SVM#pickle.loads(SVM)
        #else:
        #    self.SVM=None
        self.bandwidth=int(bandwidth)

        if train==1 and os.path.exists(self.cat_traindata_path):
            self.load_samples()
        #self.training_data= [[],[]]     #All kps of the all samples in the category
        


    def load_samples(self):
        log('Reading Training data for ' + self.name +' ...')
        self.training_pos_input_path=os.path.join(self.cat_traindata_path,'pos')
        self.training_neg_input_path=os.path.join(self.cat_traindata_path,'neg')
        self.training_pos_c_input_path=os.path.join(self.cat_traindata_path,'pos_c')
        self.training_neg_c_input_path=os.path.join(self.cat_traindata_path,'neg_c')
        self.training_gt_input_path=os.path.join(self.cat_traindata_path,'gt')

        if os.path.exists(self.training_pos_input_path):
            for file in os.listdir(self.training_pos_input_path):
                #log('Reading Positive Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="png"): 
                    s = Sample(file,self.training_pos_input_path)
                    self.samplesPos.append(s)


        if os.path.exists(self.training_pos_c_input_path):
            for file in os.listdir(self.training_pos_c_input_path):
                log('Reading Positive Combosite Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="png"): 
                    gt_filename=os.path.join(self.training_gt_input_path,filename + '.txt')
                    f = open(gt_filename, "r")
                    for line in f.readlines():
                        if line != '\n' and line !='':
                            l=line.split(',')
                            del l[-1]
                            for i in range (4):
                                l[i]=int(l[i].strip('()'))
                    
                            s = Sample(file,self.training_pos_c_input_path,l)
                        self.samplesPos.append(s)   
        log('Read Positive Samples: ' + str(len(self.samplesPos)) + ' file(s)')

        if self.bandwidth==0 or self.bandwidth == None:
            self.__calculate_bandwidth()
        
        if os.path.exists(self.training_neg_input_path):
            for file in os.listdir(self.training_neg_input_path):
                #log('Reading Negative Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="png"): 
                    s = Sample(file,self.training_neg_input_path)
                    self.samplesNeg.append(s)

        if os.path.exists(self.training_neg_c_input_path):
            for file in os.listdir(self.training_neg_c_input_path):
                #log('Reading Negative Combosite Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="png"): 
                    #gt_filename=os.path.join(self.training_gt_input_path,filename + '.txt')
                    #f = open(gt_filename, "r")
                    #for line in f.readlines():
                    #    if line != '\n' and line !='':
                    #        l=line.split(',')
                    #        del l[-1]
                    #        for i in range (4):
                    #            l[i]=int(l[i].strip('()'))
                    b=self.bandwidth
                    img=cv2.imread(os.path.join(self.training_neg_c_input_path,file),cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
                    if gray:
                        rows,cols=img.shape
                    else:
                        rows,cols,ch=img.shape
                    cc=np.array([[i,j] for i in range(int(b/2),cols,b) for j in range(int(b/2),rows,b)])#Get centers of patches
                    for c in cc:#for each patch center
                        l=[c[0]-b/2,c[1]-b/2,c[0]+b/2,c[1]+b/2]#get left up corener and bottom righ corner coords
                        s = Sample(file,self.training_neg_c_input_path,l)
                        self.samplesNeg.append(s)   
        log('Read Negative Samples: ' + str(len(self.samplesNeg)) + ' file(s)')


    def __calculate_bandwidth(self):
        #widths=[]
        #heights=[]
        widthsM=[]
        heightsM=[]
        log('Calculating bandwidth ...')
        assert "No samples or training data",(len(self.samplesPos)>0 and self.bandwidth==0)
        for s in self.samplesPos:
            #widths.append(s.width)
            #heights.append(s.height)
                
            try:
                wM,hM=s.get_geo_dims()
            except:
                s.calc_dimensions()
                wM,hM=s.get_geo_dims()

            widthsM.append(wM)
            heightsM.append(hM)
        if len(self.samplesPos) > 0:
            self.max_widthM=np.mean(widthsM)    
            self.max_heightM=np.mean(heightsM)
        else:
            self.max_widthM=0 
            self.max_heightM=0

        self.bandwidth=int(sqrt(pow(self.max_heightM,2)+pow(self.max_widthM,2))*0.75)
        log(self.name + ' Bandwidth = ' + str(self.bandwidth))

################################################################
##########################Train###############################
#################################################################     
    def train3(self,algorithm,retrain=1,reduced=0):#
        log(self.name + ': Loading training data from DB ...')
        self.training_data_pos,self.training_data_neg,self.vocab=self.loadTrainData2(algorithm)
        self.training_data_all=np.concatenate((self.training_data_pos,self.training_data_neg), axis=0)
        
        
        log(self.name + ': training data pos found in db: ' + str(len(self.training_data_pos)))
        log(self.name + ': training data Neg found in db: ' + str(len(self.training_data_neg)))
        log(self.name + ': training vocabulary found in db: ' + str(len(self.vocab)))
        if self.SVM is not None:
            log(self.name + ': SVMs found in db. ')
        
#######################################
        if len(self.vocab)==0:
            if len(self.samplesPos)==0:
                self.load_samples()
                if len(self.samplesPos)==0:
                    raise('No Trainng Data')
                    return
            self.calculate_train_data(algorithm)
            if reduced:
                self.reduce_training_data(reduced)
            self.build_vocab(algorithm)
            self.build_training_hitsograms(algorithm)
            
            self.train_classifier(algorithm)
        else:
            if len(self.samplesPos)>0:
                if retrain:# and query_yes_no('Training Vocabulary found, Construct Again?','no'):
                    if retrain ==1:
                        self.calculate_train_data(algorithm)
                        if reduced:
                            self.reduce_training_data(reduced)
                        self.build_vocab(algorithm)
                        self.build_training_hitsograms(algorithm)
         
                    self.train_classifier(algorithm)

    

    def calculate_train_data(self,algorithm):
        if len(self.samplesPos)==0:
            self.load_samples()

        log(self.name + ': Training samples...')
        sp_count=len(self.samplesPos)
        sn_count=len(self.samplesNeg)
        #if len(self.training_data_pos)>0 and query_yes_no('Training data found, pos='+str(len(self.training_data_pos))+',Neg='+str(len(self.training_data_neg))+ ',\n Construct Again?','no'):
        self.training_data_pos=[]
        self.training_data_neg=[]
        i=1
        for s in self.samplesPos:
            #log('Training sample Pos ' + str(i) +' of ' + str(sp_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            s.kps=algorithm.get_kps(img)
            #plot_features2(img,s.kps[0],savefig=1,imagePath='e:\\master\\data\\temp\\p_'+ str(i) + s.file_name ,title='Pos Sample KPS')
            img=None
            if s.kps[0] != None and s.kps[0] != []:
                self.training_data_pos.extend(s.kps[1])
        log('Positive Features Collected : ' + str(len(self.training_data_pos)))    
        
    #################################
        i=1
        for s in self.samplesNeg:
            #log('Training sample Neg ' + str(i) +' of ' + str(sn_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            s.kps=algorithm.get_kps(img)
            #plot_features2(img,s.kps[0],savefig=1,imagePath='e:\\master\\data\\temp\\n_'+ str(i) + s.file_name ,title='Neg Sample KPS')
            img=None
            if len(s.kps[0]) > 0 :
                self.training_data_neg.extend(s.kps[1])
        
        log('Negative Features Collected : ' + str(len(self.training_data_neg)))   
              
##################################         
        self.training_data_all=np.concatenate((self.training_data_pos,self.training_data_neg), axis=0)
        
        
    def build_vocab(self,algorithm):
        #Build Vocanulary

        log(self.name + ': Building Vocabulary')
        
        if len(self.training_data_all)==0:
            self.calculate_train_data(algorithm)


        if algorithm.k<1:
            K=int(algorithm.k*len(self.training_data_all))
        else:
            K=algorithm.k
        bowTrainer=cv2.BOWKMeansTrainer(K)
        

        self.vocab=self.clusterTrainData(self.training_data_all,K)
        #self.vocab=bowTrainer.cluster(np.array(self.training_data_all))
        

        #algorithm.bowextractor.setVocabulary(self.vocab)
    ###################################################               
        #log('Saving Vocabulary')
        #self.saveTrainData(algorithm)

    def build_training_hitsograms(self,algorithm):
        #if self.SVM is None or query_yes_no('SVM found, Construct Again?','no'):
        log('Building Histogram')
        #Build Histograms
        
        if self.SVMTrainData==[]:#delete SVMTrainData to force regenerate histograms
            self.SVMTrainLabel=[]
            self.SVMTrainData=[]
        
            if len(self.samplesPos) == 0:
                self.load_samples()
            
            sp_count=len(self.samplesPos)
            sn_count=len(self.samplesNeg)
        
            if len(self.vocab)==0:
                self.build_vocab(algorithm)

            #algorithm.bowextractor.setVocabulary(self.vocab)
            i=1
            for s in self.samplesPos:
                log('Generate Pos Histogram of ' + str(i) +' of ' + str(sp_count) + ' : ' + s.file_name)
                i+=1
                img=s.get_img()
                if s.kps==[] or s.kps[0]==[]:
                    s.kps=algorithm.get_kps(img)
                if s.kps[1] != None and s.kps[1] != []:
                    s.hist=algorithm.get_hist(img,s.kps[0],s.kps[1],self.vocab)
                    #s.hist=algorithm.bowextractor.compute(s.kps[1],self.vocab)
                    self.SVMTrainLabel.extend([1])
                    self.SVMTrainData.extend(s.hist)
                else:
                    hist=np.zeros(len(self.vocab), dtype=np.float32)
                    hist=hist.reshape(1,len(self.vocab))
                img=None
            i=1
            b=self.bandwidth
            for s in self.samplesNeg:
                log('Generate Neg Histogram of ' + str(i) +' of ' + str(sn_count) + ' : ' + s.file_name)
                i+=1
                img=s.get_img()
                if s.kps==[] or s.kps[0]==[]:
                    s.kps=algorithm.get_kps(img)
                if s.kps[0] != None and s.kps[0] != []:
                    s.hist=algorithm.get_hist(img,s.kps[0],s.kps[1],self.vocab)
                    #s.hist=algorithm.bowextractor.compute(s.kps[1],self.vocab)
                    self.SVMTrainLabel.extend([0])
                    self.SVMTrainData.extend(np.array(s.hist))
                else:
                    hist=np.zeros(len(self.vocab), dtype=np.float32)
                    hist=hist.reshape(1,len(self.vocab))
                img=None
                    
            self.y_train=np.array(self.SVMTrainLabel)
            self.X_train=np.array(self.SVMTrainData)
        log('Histograms Calculated')

    def train_classifier(self,algorithm):        
        if self.SVMTrainData == []:
            self.build_training_hitsograms(algorithm)
            
        log('Building PCA...')
        self.X_train=np.array(self.SVMTrainData)
        self.y_train=np.array(self.SVMTrainLabel)        
        if algorithm.n>0:
            self.PCA=RandomizedPCA(algorithm.n, whiten=True).fit(self.X_train)
            self.X_train = self.PCA.transform(self.X_train)
            
        log('Training Classifier...')
        self.SVM = svm.SVC(probability=True)
        self.SVM.fit(self.X_train,self.y_train)
        log('Classifier Trained')

        self.saveSVM()
###################################################    

    def clusterTrainData(self,data,factor=0.1):
        if factor<1:
            k=int(len(data)*factor)
        else:
            k=factor
        if k>0 and len(data)>0:
            est=KMeans(k,init_size=k*4)
            log('Clustering Composite features...')
            est.fit(data)
            labels = est.labels_
            reduced_data=est.cluster_centers_
            return reduced_data.astype(np.float32)
        else:
            #return np.array([], dtype=np.float32).reshape(0,128)
            return None

    def saveTrainData_(self,algorithm):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        conn.execute("delete from vocabData where category=? and algorithm=?",(self.id,algorithm.id))
        conn.execute("delete from TrainingData where category=? and algorithm=?",(self.id,algorithm.id))
        
        for des in self.training_data_pos:
            conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,1)",
                         (des, self.id, algorithm.id))
        for des in self.training_data_neg:
            conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,0)",
                         (des, self.id, algorithm.id))

        for v in self.vocab:
            conn.execute("insert into vocabData (descriptor,category,algorithm)values(?,?,?)",
                            (v, self.id, algorithm.id))
        s = pickle.dumps(self.SVM)
        p = pickle.dumps(self.PCA)

        conn.execute("update category set svm=?, bandwidth=?, pca=? where id=?", (s,self.bandwidth,p, self.id))
        
        conn.commit()
        conn.close()   

    def saveTrainData2(self,algorithm):
        import copy
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        s = pickle.dumps(self.SVM)
        p = pickle.dumps(self.PCA)
        v = pickle.dumps(self.vocab)
        tp = pickle.dumps(self.training_data_pos)
        tn = pickle.dumps(self.training_data_neg)

        conn.execute("insert into classifier (category,algorithm,SVM,PCA,traindata_pos,traindata_neg,vocab)values(?,?,?,?,?,?,?)",
                         ( self.id, algorithm.id,s,p,tp,tn,v))
        c=copy.deepcopy(self)
        for s in c.samplesPos:
            if len(s.kps)>0:
                s.kps[0][:]=[]
        for s in c.samplesNeg:
            if len(s.kps)>0:
                s.kps[0][:]=[]
        cs = pickle.dumps(c)
        
        conn.execute("update category_pickle set enabled=0 where category=? and algorithm=?",(self.id,algorithm.id))
        conn.execute("insert into category_pickle (category,algorithm,pickle,pca,coding) values(?,?,?,?,?)", (self.id,algorithm.id,cs,algorithm.n,bow_encoding))
        conn.commit()
        conn.close()   
        
                    
    def saveSVM(self):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        s = pickle.dumps(self.SVM)
        p = pickle.dumps(self.PCA)

        conn.execute("update category set svm=?, bandwidth=?, pca=? where id=?", (s,self.bandwidth,p, self.id))                
        conn.commit()
        conn.close()     
           
    def savePCA(self):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        s = pickle.dumps(self.PCA)
        conn.execute("update category set pca=? where id=?", (s,self.id))
        conn.commit()
        conn.close() 
                   
    def loadTrainData2(self,algorithm,):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

        cur = conn.cursor()
        descsPos=[]
        descsNeg=[]
        Vocab=[]
        
       # if localizer!=LOCALIZER_WINDOW:
		#Load Positive descriptors
        cur.execute("SELECT descriptor  from trainingData t join category c on (t.category=c.id and t.algorithm=c.algorithm) where c.id = ? and c.algorithm=? and positive=1" ,(self.id , algorithm.id))
        for r in cur:
            descsPos.append(r[0])
        
        #Load Negative descriptors
        cur.execute("SELECT descriptor  from trainingData t join category c on  (t.category=c.id and t.algorithm=c.algorithm) where c.id = ? and c.algorithm=? and positive=0" ,(self.id , algorithm.id))
        for r in cur:
            descsNeg.append(r[0])
          
        ##Load BOW Vocabulary
        cur.execute("SELECT descriptor from vocabData t join category c on (t.category=c.id and t.algorithm=c.algorithm) where c.id = ? and c.algorithm=?" ,(self.id , algorithm.id))
        for r in cur:
            desc=np.array(r[0])
            Vocab.append(desc)

        conn.close()
        return descsPos,descsNeg,np.asarray(Vocab,dtype=np.float32)

    def match_features(self,des1,des2,algorithm,accuracy = 0.75):#BOW
        if des1 is None or des1==[] or des2 is None or des2==[]:
            return []
            
        good = []
        if len(des1)>0 and len(des2)>0:
            matches = algorithm.matcher.knnMatch(des1,np.asarray(des2,np.float32),k=2)
            for m,n in matches:
                if m.distance < accuracy * n.distance:
                    good.append(m)
        
        return good                    

    def get_detection_stats(self,TT,CC):
        bandwidth=self.bandwidth
        tpos =0
        fpos =0
        fneg=len(TT)
        fecho=0
        matched=np.array(np.zeros(len(TT)))
        if len(CC)>0:
            for cc in CC:
                i,d = find_nearest_position(cc,TT)
                if d < bandwidth:
                    #print ('Point ', cc, ' --> Nearest Point : ', TT[i], ' @ Distance : ', d)
                    matched[i]+=1
                    if matched[i]==1:
                        tpos+=1
                        fneg-=1
                    else:
                        fecho+=1
                else:
                    #print ('Point ', cc, ' --> No Match, Min distance found : ', d)
                    fpos+=1
         
            precision = (tpos/(tpos+fpos))
            recall = tpos/(tpos+fneg) 
        else:
            precision,recall=0,0

        return [precision,recall,tpos,fpos,fneg,fecho]

    def get_detection_stats2(self,TT,CC):
        bandwidth=self.bandwidth
        tpos =0
        fpos =0
        fneg=len(TT)
        fecho=0
        matched=np.array(np.zeros(len(TT)))
        if len(CC)>0:
            for cc in CC:
                i,d = find_nearest_position2(cc,TT)
                if d < bandwidth:
                    #print ('Point ', cc, ' --> Nearest Point : ', TT[i], ' @ Distance : ', d)
                    matched[i]+=1
                    if matched[i]==1:
                        tpos+=1
                        fneg-=1
                    else:
                        fecho+=1
                else:
                    #print ('Point ', cc, ' --> No Match, Min distance found : ', d)
                    fpos+=1
         
            precision = (tpos/(tpos+fpos))
            recall = tpos/(tpos+fneg) 
        else:
            precision,recall=0,0

        return [precision,recall,tpos,fpos,fneg,fecho]
    

    def generate_reduced_pos_training_data(self,k_):
        if k_<1:
            kp=int(k_*len(self.training_data_pos))
            kn=int(k_*len(self.training_data_neg))
        else:
            kp=kn=k_
    
        self.reduced_training_data_pos=self.clusterTrainData(self.training_data_pos,kp)#Reduced Training data is descriptors only. no KP
        log('Positive Features Reduced to ' + str(len(self.reduced_training_data_pos)))
        
    def saveTrainDataFull(self,algorithm):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        try:
            conn.execute("delete from TrainingData where category=? and algorithm=?",(self.id,algorithm.id))
        
            for des in self.training_data_pos:
                conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,1)",
                             (des, self.id, algorithm.id))
            for des in self.training_data_neg:
                conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,0)",
                             (des, self.id, algorithm.id))
            conn.commit()
        except:
            conn.rollback()
        finally:
            conn.close()               

#    def calculate_train_data(self,algorithm):
#        log('Training samples.')
#        sp_count=len(self.samplesPos)
#        sn_count=len(self.samplesNeg)
#        i=1
#        for s in self.samplesPos:
#            #log('Training sample Pos ' + str(i) +' of ' + str(sp_count) + ' : ' + s.file_name)
#            i+=1
#            img=s.get_img()
#            s.kps=algorithm.get_kps(img)
#            #plot_features2(img,s.kps[0],savefig=1,imagePath='e:\\master\\data\\temp\\p_'+ str(i) + s.file_name ,title='Pos Sample KPS')
#            img=None
#            if s.kps[0] != None and s.kps[0] != []:
#                self.training_data_pos[0].extend(s.kps[0])
#                self.training_data_pos[1].extend(s.kps[1])
            
#        self.reduced_training_data_pos=self.clusterTrainData(self.training_data_pos[1],0.1)#Reduced Training data is descriptors only. no KP
#        log('Positive Features Collected : ' + str(len(self.training_data_pos[0])) 
#            + ', Reduced to ' + str(len(self.reduced_training_data_pos)))
##################################
#        i=1
#        for s in self.samplesNeg:
#            #log('Training sample Neg ' + str(i) +' of ' + str(sn_count) + ' : ' + s.file_name)
#            i+=1
#            img=s.get_img()
#            s.kps=algorithm.get_kps(img)
#            #plot_features2(img,s.kps[0],savefig=1,imagePath='e:\\master\\data\\temp\\n_'+ str(i) + s.file_name ,title='Neg Sample KPS')
#            img=None
#            if len(s.kps[0]) > 0 :
#                self.training_data_neg[0].extend(s.kps[0])
#                self.training_data_neg[1].extend(s.kps[1])
            
#        self.reduced_training_data_neg=self.clusterTrainData(self.training_data_neg[1],0.1)#Reduced Training data is descriptors only. no KP
#        log('Negative Features Collected : ' + str(len(self.training_data_neg[0])) 
#            + ', Reduced to ' + str(len(self.reduced_training_data_neg)))            
###################################         
#        self.training_data_all=np.concatenate((self.reduced_training_data_pos,self.reduced_training_data_neg), axis=0)
#        self.training_data_all_label=[]
#        self.training_data_all_label.extend(np.ones(len(self.reduced_training_data_pos)))
#        self.training_data_all_label.extend(np.zeros(len(self.reduced_training_data_neg)))
#        #self.training_data_all=self.reduced_training_data_pos.copy()
#        #self.training_data_all.extend(self.reduced_training_data_neg)
#        #self.training_data_all=self.reduced_training_data_pos+self.reduced_training_data_neg
######################################            
def cluster_features_meanshift(kp,good,src_pts,bandwidth=0):
    if len(src_pts)>0:
        if bandwidth==0:
            bandwidth = estimate_bandwidth(src_pts, quantile=0.2, n_samples=len(src_pts))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(src_pts)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        return cluster_centers,labels   
    else:
        return [],[]

def cluster_features_window(kp,good,src_pts,bandwidth=0):
    if len(src_pts)>0:
        if bandwidth==0:
            bandwidth = estimate_bandwidth(src_pts, quantile=0.2, n_samples=len(src_pts))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(src_pts)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        return cluster_centers,labels   
    else:
        return [],[]

def cluster_features_dbscan(kp,good,src_pts,eps_=.2,min_samples_=1,bandwidth=50):
    from sklearn.preprocessing import StandardScaler
    # Compute DBSCAN
    if len(src_pts) == 0:
        return [],[]
    g = StandardScaler().fit_transform(src_pts)
    dbscan = DBSCAN(eps=eps_, min_samples=min_samples_).fit(g)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_centers=[]
    #cc_f=[]
    for k in range(n_clusters_):
        my_members = labels == k
        #members_count=np.count_nonzero(my_members)
        #dx=0
        #dy=0
        #dx=sum(src_pts[my_members, 0])/members_count
        #dy=sum(src_pts[my_members, 1])/members_count
        
        (x,y),radius = cv2.minEnclosingCircle(src_pts[my_members])
        
        
        if radius < bandwidth*2:
            cluster_centers.append([x,y])
        
    return np.array(cluster_centers),labels


def loadCategories_pickle(algorithm):
    conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

    cats=[]
    cur = conn.cursor()
    cur.execute("select pickle from category_pickle p where p.algorithm=? and p.enabled=1 order by category",(algorithm.id,))

    for r in cur:
        c=r[0]
        cats.append(c)
    return cats


def loadCategories(train=1):
    conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

    cats=[]
    cur = conn.cursor()
    cur.execute("select id,name,bandwidth,SVM,PCA from category ")
    for r in cur:
        c=Category(r[0],r[1],r[2],r[3],r[4],train=train)
        cats.append(c)
    return cats

def load_categories_from_file(path):
    return [load_category_from_file('airplane',path),
            load_category_from_file('airport',path),
            load_category_from_file('car',path)]


def load_category_from_file(category_name,path):
    filename=category_name + '.dat'
    file=open(os.path.join(path,filename),'rb')
    category=pickle.load(file)
    file.close()
    return category


def save_category_to_file(category,outdir):
    filename=category.name + '.dat'
    file=open(os.path.join(outdir,filename),'wb')
    c=copy.deepcopy(category)
    for s in c.samplesPos:
        if len(s.kps)>0:
            s.kps[0][:]=[]
    for s in c.samplesNeg:
        if len(s.kps)>0:
            s.kps[0][:]=[]
    pickle.dump(c,file)
    c=None
    file.close()

    
def save_all_categories(categories,outdir):
    
    for category in categories:
        save_category_to_file(category,outdir)


def beeb():
    import winsound
    Freq = 2500 # Set Frequency To 2500 Hertz
    Dur = 300 # Set Duration To 1000 ms == 1 second
    winsound.Beep(Freq,Dur)

def save_data():
    import pickle
    file=open(os.path.join(out_dir,'y_test.dat'),'wb')
    pickle.dump(c_y_tests,file)
    file.close()
    file=open(os.path.join(out_dir,'y_score.dat'),'wb')
    pickle.dump(c_y_scores,file)
    file.close()    
    lib2.save_all_categories(categories,out_dir)       


def test(categories,algorithm):
    startTime=datetime.now()
    op_id='{0}_PCA{1}_BOW{2}_ALLData_Color_Hard_localizer{5}_skCluster_reduced{3}_{4}_accuracy_{6}_coding_{7}'
    op_id=op_id.format(algorithm.detector_name,algorithm.n,algorithm.k,reduced,startTime.strftime('%Y%m%d_%H%M%S'),localizer,accuracy,bow_encoding)  
    out_dir_test=os.path.join(out_dir,op_id)
    os.makedirs(out_dir_test, exist_ok=True)
    log_file = open(os.path.join(out_dir_test , 'output.log'), 'w')
    
    notes=''
    notes+= 'Testing with detector='+ algorithm.detector_name 
    notes+= ', gray=' + str(gray) + ', vocab=' + str(algorithm.k) 
    notes+= ', pca with n=' + str(algorithm.n) 
    notes+=', localizer = ' + str(localizer) 
    if localizer==0:
        notes+= ', window scanning, window_size_factor=' + str(window_size_factor) 
        notes+=',window_overlapping_factor=' + str(window_overlapping_factor) 
    notes+=' \n, Usin Coding: ' +  str(bow_encoding)
    if reduced >0:
        notes+=' Data reduced with k=' + str(reduced)

    log(notes,log_file)

    c_y_tests=[] 
    c_y_scores=[]
    c_results=[]
    times=[]
    
    for category in categories:
        Trainingdir_out = os.path.join(out_dir_test , 'Training_out',category.name)
        Testingdir_out = os.path.join(out_dir_test , 'Testing_out',category.name)
        os.makedirs(Trainingdir_out, exist_ok=True)
        os.makedirs(Testingdir_out, exist_ok=True)
        testpath=category.cat_testdata_path
        gtpath=os.path.join(testpath,'gt')
        i=0
        filenames=[]
        filepaths=[]
        gtfilepaths=[]
        y_tests=[]
        y_scores=[]
        results=[0,0,0,0,0,0]
        if not os.path.exists(testpath):
            continue
        for testfile in os.listdir(testpath):    #ARG1
            testFile_path=os.path.join(testpath, testfile)
            if os.path.isfile(testFile_path):
                filename,extension = testfile.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"):
                    filenames.append(filename)
                    filepaths.append(testFile_path)
                    gtfile=os.path.join(gtpath,filename) + '.txt'
                    gtfilepaths.append(gtfile)
        
        for filename,testFile_path,gtfile in zip(filenames,filepaths,gtfilepaths):
            t=datetime.now()
            goodMatchs=[]
            good_kps=[]                        

            cluster_labels=[]
            i+=1

            log('Start Testing tile ' + str(i) + '/' + str(len(filenames)) + ' : ' + testFile_path,log_file)
            img=cv2.imread(testFile_path,cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)

            band=category.bandwidth*window_size_factor  #WIndows size r
            o=band*window_overlapping_factor    #Windows overlapping                         
            step=band-o
            

            #########################################################################
            ##########################Localize test points###########################
            #########################################################################
            if localizer==LOCALIZER_WINDOW:
                if gray:
                    rows,cols=img.shape
                else:
                    rows,cols,ch=img.shape
                
                log("Using Window(" + str(band) + "," + str(o) + ") Scan to Localize Targets...")
                
                log("Using Window(" + str(band) + "," + str(o) + ") Scan to Localize Targets...",log_file)

                cluster_centers=np.array([np.array([i,j]) for j in range(int(band/2),rows,int(step))
                                 for i in range(int(band/2),cols,int(step))])
            else:
                test_kps=algorithm.get_kps(img)
                if reduced > 0:
                    if (len(category.reduced_training_data_pos)!=reduced and len(category.reduced_training_data_pos)!=int(reduced*len(category.training_data_pos))):
                        category.reduced_training_data_pos=category.clusterTrainData(category.training_data_pos,reduced)                
                    goodMatchs=category.match_features(test_kps[1],category.reduced_training_data_pos,algorithm,accuracy)
                else:
                    goodMatchs=category.match_features(test_kps[1],category.training_data_pos,algorithm,accuracy)

                log('good matches = '+ str(len(goodMatchs)),log_file)
                
                plot_features2(img,test_kps[0],graph,
                                 os.path.join(Testingdir_out,'detected_features_'+ filename + '.png'),'Detcted features in image')


                for g in goodMatchs:
                    good_kps.append(np.float32(test_kps[0][g.queryIdx].pt)) #to be used in clustering 
                good_kps=np.asarray(good_kps)
                

                    
                if localizer==LOCALIZER_DBSCAN:
                    log("Using DBScan to Localize Targets...")
                    cluster_centers,cluster_labels=cluster_features_dbscan(test_kps,goodMatchs,good_kps,bandwidth=category.bandwidth,eps_=epsilon,min_samples_=min_pts_per_cluster)
                elif localizer==LOCALIZER_MEANSHIFT:
                    log("Using Mean Shift to Localize Targets...")
                    cluster_centers,cluster_labels=cluster_features_meanshift(test_kps,goodMatchs,good_kps,category.bandwidth/2)

                plot_features_and_cluster_centers(img,cluster_centers,None,cluster_labels,good_kps, graph,
                                        os.path.join(Testingdir_out,'good_features_clustered_'+ filename + '.png'),'Good features clustered in image')
            #########################################################################
            #########################Test Candidate Objects##########################
            #########################################################################
            cc_f=[]
            SVMTestLabel=[]

            TT=load_truth_table2(gtfile)
            matched=np.array(np.zeros(len(TT)))

            plot_features_and_cluster_centers(img,TT,None,[],[], graph,
                                        os.path.join(Testingdir_out,'TT_'+ filename + '.png'),'Ground Truth')

            #algorithm.bowextractor.setVocabulary(category.vocab)

            ###########Build Test data ground truth################                        
            for c in cluster_centers:
                a,d = find_nearest_position3(c,TT,matched)
                if d < band:
                    SVMTestLabel.extend([1])
                    matched[a]=1
                else:
                    SVMTestLabel.extend([0])
            ##########Add non matched positions#################
            #if np.sum(matched)<len(matched):
            #    cluster_centers2=[]
            #    for ti,mm in zip(TT,matched):
            #        if mm==0:
            #            cluster_centers2.append(ti)
            #            SVMTestLabel.extend([1])
            #    if len(cluster_centers)>0 :
            #        cluster_centers=np.concatenate((cluster_centers,np.array(cluster_centers2)),axis=0)
            #    else:
            #        cluster_centers=np.array(cluster_centers2)

            ############Build Test Data################   
            SVMTestData=[]
            for c in cluster_centers:
                b2=int(band/2)
                patch=img[(c[1]-b2 if c[1]>b2 else 0):c[1]+b2,c[0]-b2 if c[0]>b2 else 0:c[0]+b2]     #TODO may need enlargement

                #patch_kps=algorithm.detector.detect(patch)
                patch_kps,patch_descs=algorithm.descriptor.detectAndCompute(patch,None)
                #plot_features2(patch,patch_kps,0)
                if patch_descs is not None:
                    hist=algorithm.get_hist(patch,patch_kps,patch_descs,category.vocab)
                else:
                    hist=np.zeros(len(category.vocab), dtype=np.float32)
                    hist=hist.reshape(1,len(category.vocab))
                patch=None                  
                if hist is None:
                    hist=np.zeros(len(category.vocab), dtype=np.float32)
                    hist=hist.reshape(1,len(category.vocab))

                SVMTestData.extend(hist)

            X_test=np.array(SVMTestData)
            y_test=np.array(SVMTestLabel)     
            ##########################
            if X_test.shape[0]==0:
                X_test=np.zeros(len(category.vocab)).astype(np.float32)
                X_test=X_test.reshape(1,len(category.vocab))
                y_test=[0]
            if algorithm.n>0:
                X_test=category.PCA.transform(X_test)
            ##########################

            score=category.SVM.score(X_test,y_test)
            df=category.SVM.decision_function(X_test)
            
           
            if df.shape[0]>1:
                fpr, tpr, thresholds = roc_curve(y_test, df)
                roc_auc = auc(fpr, tpr)

                log('Score = ' + str(score),log_file)
                log('AUC = ' + str(roc_auc),log_file)


                precision, recall, _ = precision_recall_curve(y_test,df)
                average_precision = average_precision_score(y_test, df)

                log('average precision = ' + str(average_precision),log_file)

                y_tests.extend(y_test)
                y_scores.extend(df)

                if graph:
                    plt.plot(fpr, tpr, lw=1, label='ROC  (area = %0.2f)' % (roc_auc,))
                    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
                    plt.xlim([-0.05, 1.05])
                    plt.ylim([-0.05, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver operating characteristic')
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join(Testingdir_out,filename + '_roc.png'),bbox_inches='tight')
                    if graph>1:
                        plt.show()
                    else:
                        plt.close()


                if graph:
                    plt.clf()
                    plt.plot(recall, precision, label='Precision-Recall curve')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join(Testingdir_out,filename + '_pr.png'),bbox_inches='tight')
                    if graph>1:
                        plt.show()
                    else:
                        plt.close()


            cc_f=category.SVM.predict(X_test)

            detected_targets_for_file = pixelToLatLon(testFile_path,cluster_centers,cc_f)        
            results_for_file=category.get_detection_stats2(TT,detected_targets_for_file)
            results = [results[i] + results_for_file[i] for i in range(6)]
            
            log('Precision : '+ str(results_for_file[0]),log_file)
            log('recall : '+ str(results_for_file[1]),log_file)
            log('True Positives: '+ str(results_for_file[2]),log_file)
            log('False Positives: '+ str(results_for_file[3]),log_file)
            log('False Negatives:'+ str(results_for_file[4]),log_file)
            log('False Echo:'+ str(results_for_file[5]),log_file)
 
            log('Detected Objects all:  ' + str(len(cluster_centers)),log_file)
            log('Filtered Detected Objects:  ' + str(np.sum(cc_f)),log_file)
            log('True Objects:  ' + str(len(TT)),log_file)

            #Create images of results and save it(graph -> 0: No Graphs, 1: create and save but no show, 2: create and save and show)
            if graph:
                plot_features_and_cluster_centers(img,cluster_centers,cc_f,cluster_labels,good_kps, graph,
                                                    os.path.join(Testingdir_out,filename + '_filtered_objects.png'),'Detected classifier objects')
            img=None
            times.append(datetime.now()-t)
        #####Total PR per Category##############
        if y_tests !=[]:
            y_tests=np.array(y_tests)
            y_scores=np.array(y_scores)

            fpr, tpr, thresholds = roc_curve(y_tests.ravel(), y_scores.ravel())
            roc_auc = auc(fpr, tpr) 
            if graph:
                plt.plot(fpr, tpr, lw=1, label=category.name + ' ROC  (area = %0.2f)' % (roc_auc,))
                plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(category.name + ' Receiver operating characteristic')
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(Testingdir_out,category.name + '_roc.png'),bbox_inches='tight')
                if graph>1:
                    plt.show()
                else:
                    plt.close()


            precision, recall, _ = precision_recall_curve(y_tests.ravel(), y_scores.ravel())
            average_precision = average_precision_score(y_tests, y_scores, average="micro")
            if graph:
                plt.clf()
                plt.plot(recall, precision, label=category.name + ' Precision-Recall curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(category.name + ' Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(Testingdir_out,category.name + '_pr.png'),bbox_inches='tight')
                if graph>1:
                    plt.show()
                else:
                    plt.close()

        file=open(os.path.join(out_dir_test,category.name + '_y_test.dat'),'wb')
        pickle.dump(y_tests,file)
        file.close()
        file=open(os.path.join(out_dir_test,category.name + '_y_scores.dat'),'wb')
        pickle.dump(y_scores,file)
        file.close()
        
        c_y_tests.extend(y_tests)
        c_y_scores.extend(y_scores)            

        if results[2]+results[3] > 0 :
            results[0] = (results[2]/(results[2]+results[3]))
        else:
            results[0] = 0
        results[1] = results[2]/(results[2]+results[4])         
        c_results.append(results)
        
        log(category.name + ' Precision : '+ str(results[0]),log_file)
        log(category.name + ' recall : '+ str(results[1]),log_file)
        log(category.name + ' True Positives: '+ str(results[2]),log_file)
        log(category.name + ' False Positives: '+ str(results[3]),log_file)
        log(category.name + ' False Negatives:'+ str(results[4]),log_file)
        log(category.name + ' False Echo:'+ str(results[5]),log_file)

        endTime=datetime.now()

    
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        startTime_=startTime.strftime('%Y%m%d_%H%M%S.%f')
        endTime_=endTime.strftime('%Y%m%d_%H%M%S.%f')
        timedelta=(endTime-startTime).total_seconds()

        sql='insert into OPS_OUT '
        sql+='(start_time,end_time,epsilon,min_pts_per_cluster,accuracy,precision,recall,time,tpos,fpos,fneg,echo,algorithm,bandwidth,notes,category,localizer,wsf,wof,reduced_k,y_test,y_score,op_id_txt,soft_encoding) '
        sql+='values ( ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
        cur.execute(sql,(startTime_,endTime_,epsilon,min_pts_per_cluster,
                         accuracy,results[0],results[1],timedelta,results[2],
                         results[3],results[4],results[5],algorithm.id,
                         category.bandwidth,notes,category.id,
                         localizer,window_size_factor,window_overlapping_factor,reduced,
                         pickle.dumps(y_tests),pickle.dumps(y_scores),op_id,bow_encoding))
        conn.commit()
    ##############################End of category loop#########################################
    if c_y_tests!=[]:
        c_y_tests=np.array(c_y_tests)
        c_y_scores=np.array(c_y_scores)
        fpr, tpr, thresholds = roc_curve(c_y_tests.ravel(), c_y_scores.ravel())
        roc_auc = auc(fpr, tpr) 
        log('Total All ROC AUC=' + str(roc_auc),log_file)        

        if graph:
            plt.plot(fpr, tpr, lw=1, label='All ROC  (area = %0.2f)' % (roc_auc,))
            plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('All Receiver operating characteristic')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(out_dir_test,'All_roc.png'),bbox_inches='tight')
            if graph>1:
                plt.show()
            else:
                plt.close()

        precision, recall, _ = precision_recall_curve(c_y_tests.ravel(), c_y_scores.ravel())
        average_precision = average_precision_score(c_y_tests, c_y_scores, average="micro")

        log('Total All AP=' + str(average_precision),log_file)        

        if graph:
            plt.clf()
            plt.plot(recall, precision, label='All Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('All Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(out_dir_test,'All_pr.png'),bbox_inches='tight')
            if graph>1:
                plt.show()
            else:
                plt.close()


    file=open(os.path.join(out_dir_test,'y_test.dat'),'wb')
    pickle.dump(c_y_tests,file)
    file.close()
    file=open(os.path.join(out_dir_test,'y_score.dat'),'wb')
    pickle.dump(c_y_scores,file)
    file.close()    
    


    beeb()
    log('Finished succefully',log_file)
#     return times
    log_file.close()
