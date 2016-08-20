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

#import matplotlib.cm as cm
#import pickle
#import re
#import sys
#from PyQt4 import QtGui
#import gdal,ogr,osr
#import db

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def initialize_parameteres(data_root_='E:\\Master\\Data\\',dbname="train.db"):
    global data_root,Trainingdir_in,Testingdir_in,log_file,out_dir,db_filepath, LOCALIZER_WINDOW,LOCALIZER_MEANSHIFT,LOCALIZER_DBSCAN
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

def plot_features_and_cluster_centers(img,cluster_centers,cc_f,labels,src_pts,savefig=0,img_name=''):
    
    if len(img.shape)==3 :
        b,g,r = cv2.split(img)       # get b,g,r    
        plt.imshow(cv2.merge([r,g,b]))
    else:
        plt.imshow(img)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    n_clusters_ = len(cluster_centers)
    count=0
    for k, col in zip(range(n_clusters_), colors):
        if cc_f is None or cc_f[k] > 0:
            cluster_center = cluster_centers[k]
            my_members = labels == k
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='none',
                        markeredgecolor=col, markersize=20)
            if labels!=[]:
                plt.plot(src_pts[my_members, 0], src_pts[my_members, 1], col + '.')
            count+=1

    plt.title('Number of clusters: %d' % count)
    if savefig:
        plt.savefig(img_name)
        #log('clustering result image saved')

    if savefig > 1:
        plt.show()
    plt.close()

def plot_features2(img,key_pts,savefig=1,imagePath=''):
    if len(img.shape)==3 :
        b,g,r = cv2.split(img)       # get b,g,r    
        plt.imshow(cv2.merge([r,g,b]))
    else:
        plt.imshow(img)

                        
    src_pts=[kp.pt for kp in key_pts] #get coordinates from ketpoint class 
    x, y = zip(*src_pts)
    if len(src_pts) > 0:
        plt.plot(x,y, 'r' + '.')
    plt.title('Good Features')
    if savefig:
        plt.savefig(imagePath)
        plt.close()
    else:
        plt.show()

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
        plt.savefig(imagePath)
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
            x=l[2]-l[0]+l[0]
            y=l[3]-l[1]+l[1]        
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

        self.__calc_dimensions()
        self.kps= []
 
    def get_img(self):
        img=cv2.imread(self.file_path,cv2.IMREAD_COLOR)
        if (self.aoi==[]):
            return img
        else:  
            return img[self.aoi[1]:self.aoi[3],self.aoi[0]:self.aoi[2]]

    def set_category(self,c):
        self.category = c 

    def get_path(self):
        return os.path.join(cat_in_path, file)

    def __calc_dimensions(self):
        """Calculate the real width and height of the sample"""
        gimg=self.get_img()
 		#gimg=gdal.Open(self.file_path) 
        #gt=gimg.GetGeoTransform()
        #src = osr.SpatialReference()
        #wkt=gimg.GetProjection()
        
        row,col,ch=gimg.shape
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
    def __init__(self, det_name,des_name=None,bow=False,k=1000,matcher="FLANN"):
        self.detector_name = det_name
        self.des_name = des_name

        self.get_id()
        
        #FLANN_INDEX_KDTREE = 0
        #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        #search_params = dict(checks = 50)
        #self.matcher = cv2.FlannBasedMatcher(index_params, search_params) 
 
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
                
        self.bowTrainer=cv2.BOWKMeansTrainer(k)
        self.bowextractor = cv2.BOWImgDescriptorExtractor(self.descriptor,self.matcher)




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

    def __init__(self,id,name,bandwidth=0,SVM=None):
        self.id=id
        self.name=name
        self.samplesPos=[]
        self.samplesNeg=[]   
        self.training_data_pos=([],[])
        self.training_data_neg=([],[])     
        self.cat_traindata_path=os.path.join(Trainingdir_in, self.name)
        self.cat_testdata_path=os.path.join(Testingdir_in, self.name)
        if SVM is not None:
            self.SVM=SVM#pickle.loads(SVM)
        else:
            self.SVM=None
        self.bandwidth=int(bandwidth)

        if os.path.exists(self.cat_traindata_path):
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
                log('Reading Positive Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"): 
                    s = Sample(file,self.training_pos_input_path)
                    self.samplesPos.append(s)


        if os.path.exists(self.training_pos_c_input_path):
            for file in os.listdir(self.training_pos_c_input_path):
                log('Reading Positive Combosite Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"): 
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
        
        if self.bandwidth==0 or self.bandwidth == None:
            self.__calculate_bandwidth()
        
        if os.path.exists(self.training_neg_input_path):
            for file in os.listdir(self.training_neg_input_path):
                log('Reading Negative Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"): 
                    s = Sample(file,self.training_neg_input_path)
                    self.samplesNeg.append(s)

        if os.path.exists(self.training_neg_c_input_path):
            for file in os.listdir(self.training_neg_c_input_path):
                log('Reading Negative Combosite Sample: ' + file)
                filename,extension = file.rsplit('.',1)
                if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"): 
                    #gt_filename=os.path.join(self.training_gt_input_path,filename + '.txt')
                    #f = open(gt_filename, "r")
                    #for line in f.readlines():
                    #    if line != '\n' and line !='':
                    #        l=line.split(',')
                    #        del l[-1]
                    #        for i in range (4):
                    #            l[i]=int(l[i].strip('()'))
                    b=self.bandwidth
                    img=cv2.imread(os.path.join(self.training_neg_c_input_path,file))
                    rows,cols,ch=img.shape
                    cc=np.array([[i,j] for i in range(int(b/2),cols,b) for j in range(int(b/2),rows,b)])#Get centers of patches
                    for c in cc:#for each patch center
                        l=[c[0]-b/2,c[1]-b/2,c[0]+b/2,c[1]+b/2]#get left up corener and bottom righ corner coords
                        s = Sample(file,self.training_neg_c_input_path,l)
                        self.samplesNeg.append(s)   


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

        
    def train3(self,algorithm):#
        self.reduced_training_data_pos,self.reduced_training_data_neg,self.vocab=self.loadTrainData2(algorithm)
        #log(self.vocab)
        log(self.name + ': training data pos found in db: ' + str(len(self.reduced_training_data_pos)))
        log(self.name + ': training data Neg found in db: ' + str(len(self.reduced_training_data_neg)))
        log(self.name + ': training vocabulary found in db: ' + str(len(self.vocab)))
        if self.SVM is not None:
            log(self.name + ': SVMs found in db. ')

#######################################
        if len(self.vocab)==0:
            if len(self.samplesPos)==0:
                assert('No Trainng Data')
                return
            else:
                self.train_samples()
        else:
            if len(self.samplesPos)==0:
                return
            else:
                if query_yes_no('Training Vocabulary found, Construct Again?','no'):
                    self.train_samples()

    def train_samples(self):
        log('Training samples.')
        sp_count=len(self.samplesPos)
        sn_count=len(self.samplesNeg)
        i=1
        for s in self.samplesPos:
            log('Training sample Pos ' + str(i) +' of ' + str(sp_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            s.kps=algorithm.get_kps(img)
            img=None
            if s.kps[0] != None and s.kps[0] != []:
                self.training_data_pos[0].extend(s.kps[0])
                self.training_data_pos[1].extend(s.kps[1])
            
        self.reduced_training_data_pos=self.clusterTrainData(self.training_data_pos[1],0.1)#Reduced Training data is descriptors only. no KP
        log('Positive Features Collected : ' + str(len(self.training_data_pos[0])) 
            + ', Reduced to ' + str(len(self.reduced_training_data_pos)))
#################################
        i=1
        for s in self.samplesNeg:
            log('Training sample Neg ' + str(i) +' of ' + str(sn_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            s.kps=algorithm.get_kps(img)
            img=None
            if len(s.kps[0]) > 0 :
                self.training_data_neg[0].extend(s.kps[0])
                self.training_data_neg[1].extend(s.kps[1])
            
        self.reduced_training_data_neg=self.clusterTrainData(self.training_data_neg[1],0.1)#Reduced Training data is descriptors only. no KP
        log('Negative Features Collected : ' + str(len(self.training_data_neg[0])) 
            + ', Reduced to ' + str(len(self.reduced_training_data_neg)))            
##################################         
        self.training_data_all=[[],[]]
        self.training_data_all[0]=[]
        self.training_data_all[1]=np.concatenate((self.reduced_training_data_pos,self.reduced_training_data_neg), axis=0)
#####################################            
        
    
        #Build Vocanulary

        log('Building Vocabulary')
        vocab=self.clusterTrainData(self.training_data_all[1])
            
        #vocab=algorithm.bowTrainer.cluster(np.array(self.training_data_all[1]))
            
        #Reduce Vocabulary Dimensionality with PCA
        #from sklearn import decomposition
        #pca = decomposition.PCA(n_components=64)    #TODO
        #pca.fit(vocab)
        #self.vocab = pca.transform(vocab)
        self.vocab = vocab
        algorithm.bowextractor.setVocabulary(self.vocab)
###################################################               
        log('Saving Vocabulary')
        self.saveTrainData(algorithm)
###################################################
    #if self.SVM is None or query_yes_no('SVM found, Construct Again?','no'):
        log('Building Histogram')
        #Build Histograms
        SVMTrainLabel=[]
        SVMTrainData=[]
        i=1
        for s in self.samplesPos:
            log('Generate Pos Histogram of ' + str(i) +' of ' + str(sp_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            if s.kps[0] != None and s.kps[0] != []:
                s.hist=algorithm.bowextractor.compute(img,s.kps[0])
                SVMTrainLabel.extend([1])
                SVMTrainData.extend(s.hist)
            img=None
        i=1
        b=self.bandwidth
        for s in self.samplesNeg:
            log('Generate Neg Histogram of ' + str(i) +' of ' + str(sn_count) + ' : ' + s.file_name)
            i+=1
            img=s.get_img()
            if len(s.kps[0]) > 0 :
                s.hist=algorithm.bowextractor.compute(img,s.kps[0])
                SVMTrainLabel.extend([0])
                SVMTrainData.extend(np.array(s.hist))
            img=None
        log('Histograms Calculated')

            
        log('Training Classifier')
        X_train=np.array(SVMTrainData)
        y_train=np.array(SVMTrainLabel)

        self.SVM = svm.SVC(kernel='linear', probability=True)
        self.SVM.fit(X_train,y_train)

            

        self.saveSVM()
###################################################    

    def clusterTrainData(self,data,factor=0.1):
        k=int(len(data)*factor)
        if k>0:
            est=KMeans(k,init_size=k*4)
            log('Clustering Composite features...')
            est.fit(data)
            labels = est.labels_
            reduced_data=est.cluster_centers_
            return reduced_data.astype(np.float32)
        else:
            return np.array([], dtype=np.float32).reshape(0,128)


    def saveTrainData(self,algorithm):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        conn.execute("delete from vocabData where category=? and algorithm=?",(self.id,algorithm.id))
        conn.execute("delete from TrainingData where category=? and algorithm=?",(self.id,algorithm.id))
        
        for des in self.reduced_training_data_pos:
            conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,1)",
                         (des, self.id, algorithm.id))
        for des in self.reduced_training_data_neg:
            conn.execute("insert into TrainingData (descriptor,category,algorithm,positive)values(?,?,?,0)",
                         (des, self.id, algorithm.id))

        for v in self.vocab:
            conn.execute("insert into vocabData (descriptor,category,algorithm)values(?,?,?)",
                            (v, self.id, algorithm.id))
        conn.commit()
        conn.close()   
             
    def saveSVM(self):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur=conn.cursor()

        s = pickle.dumps(self.SVM)
        conn.execute("update category set svm=?, bandwidth=? where id=?", (s,self.bandwidth,self.id))
        conn.commit()
        conn.close()        
        
    def loadTrainData2(self,algorithm):
        conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

        cur = conn.cursor()
        descsPos=[]
        descsNeg=[]
        Vocab=[]
        
        #Load Positive descriptors
        cur.execute("SELECT descriptor  from trainingData t join category c on t.category=c.id where c.id = ? and algorithm=? and positive=1" ,(self.id , algorithm.id))
        for r in cur:
            descsPos.append(r[0])
        
        #Load Negative descriptors
        cur.execute("SELECT descriptor  from trainingData t join category c on t.category=c.id where c.id = ? and algorithm=? and positive=0" ,(self.id , algorithm.id))
        for r in cur:
            descsNeg.append(r[0])
          
        #Load BOW Vocabulary
        cur.execute("SELECT descriptor from vocabData t join category c on t.category=c.id where c.id = ? and algorithm=?" ,(self.id , algorithm.id))
        for r in cur:
            desc=np.array(r[0])
            Vocab.append(desc)

        conn.close()
        return descsPos,descsNeg,np.asarray(Vocab,dtype=np.float32)

    def match_features_with_reduced_pos(self,des1,algorithm,Accuracy = 0.75):#BOW
        des2=self.reduced_training_data_pos
        if des2 != []:
            if des1 is None:
                return []
            
            good = []
            flann = algorithm.matcher
            if len(des1)>0 and len(des2)>0:
                matches = flann.knnMatch(des1,np.asarray(des2,np.float32),k=2)
                for m,n in matches:
                    if m.distance < Accuracy * n.distance:
                        good.append(m)
        
            return good                    
        else:
            return []

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
        return [],[],None
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


def loadCategories(conn):
    cats=[]
    cur = conn.cursor()
    cur.execute("select id,name,bandwidth from category ")
    for r in cur:
        c=Category(r[0],r[1],r[2])
        cats.append(c)
    return cats


def loadCategories():
    conn = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)

    cats=[]
    cur = conn.cursor()
    cur.execute("select id,name,bandwidth,SVM from category ")
    for r in cur:
        c=Category(r[0],r[1],r[2],r[3])
        cats.append(c)
    return cats