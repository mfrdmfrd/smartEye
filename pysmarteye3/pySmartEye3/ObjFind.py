# coding: utf-8

import numpy as np
import cv2
import os
import mylibrary as m
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import threading
import sqlite3
import random
#from osgeo import gdal,ogr,osr

class objfind():#threading.Thread):
    def __init__(self, algorithm, epsilon_ = 0.2,min_pts_per_cluster_=3,accuracy_=0.75,
                 category='Aircraft',i=0,bandwidth=0,dbscan=0,graph=1,geo=0,testpath=None,
                 notes='None',localizer=0,window_size_factor=2,
            window_overlapping_factor=.5): 
        #threading.Thread.__init__(self) 
        self.epsilon = epsilon_
        self.min_pts_per_cluster=min_pts_per_cluster_
        self.accuracy=accuracy_
        self.op_id=datetime.now().strftime('%Y%m%d_%H%M%S' + ('_'+str(i) if i > 0 else ''))
        self.algorithm=algorithm           #ARG3
        self.gray=0
        self.out_dir=os.path.join(m.out_dir,self.op_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.Trainingdir_out = os.path.join(self.out_dir , 'Training_out',category.name)
        self.Testingdir_out = os.path.join(self.out_dir , 'Testing_out',category.name)
        os.makedirs(self.Trainingdir_out, exist_ok=True)
        os.makedirs(self.Testingdir_out, exist_ok=True)
        self.log_file = open(os.path.join(self.out_dir , 'output.log'), 'w')
        self.category=category
        #self.clusteringDBSCAN=dbscan
        self.graph=graph
        if testpath == None:
            testpath=m.Testingdir_in
        
        self.localizer=localizer
        self.windows_size_factor=window_size_factor
        self.window_overlapping_factor=window_overlapping_factor

        self.testpath=testpath
        self.detected_targets=[]
        self.geo=geo
        self.notes=notes

    def run_bow(self):
        self.startTime=datetime.now()#.strftime('%Y%m%d_%H%M%S.%f')
        self.files=[]
        self.gtfiles=[]
        i=0 
        m.log ('Start Testing ' + self.category.name + ' With algorithm ' + self.algorithm.detector_name + ' , BOW and SVM',self.log_file)
        m.log ('With Bandwidth ' + str(self.category.bandwidth),self.log_file)

        self.results=[0,0,0,0,0,0]

        gtpath=os.path.join(self.testpath,'gt')
        if os.path.exists(self.testpath):
            conn = sqlite3.connect(m.db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
            for testfile in os.listdir(self.testpath):    #ARG1
                testFile_path=os.path.join(self.testpath, testfile)
                if os.path.isfile(testFile_path):
                    filename,extension = testfile.rsplit('.',1)
                    if (extension=="tif" or extension=="jpg" or extension=="bmp" or extension=="png"):
                        self.files.append(testfile)
                        gtfile=os.path.join(gtpath,filename) + '.txt'
                        self.gtfiles.append(gtfile)
               
                        goodMatchs=[]
                        good_kps=[]                        
                        cluster_labels=[]
                        i+=1
                        
                        m.log('Start Testing tile ' + str(i) + ' : ' + testfile,self.log_file)
                        img=cv2.imread(testFile_path,cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
                        
                        band=self.category.bandwidth*self.windows_size_factor  #WIndows size r
                        o=band*self.window_overlapping_factor    #Windows overlapping                         
                        step=band-o

                        if self.localizer==m.LOCALIZER_WINDOW:
                            m.log("Using Window(" + str(band) + "," + str(o) + ") Scan to Localize Targets...")
                            rows,cols,ch=img.shape
                            cluster_centers=np.array([np.array([i,j]) for j in range(int(band/2),rows,int(step))
                                             for i in range(int(band/2),cols,int(step))])
                        
                        else:


                            test_kps=self.algorithm.get_kps(img)
            
                            #Create images of results and save it(self.graph -> 0: No Graphs, 1: create and save but no show, 2: create and save and show)
                            #if self.graph:
                            #    img_t=cv2.drawKeypoints(img,test_kps[0],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                            #    if len(img.shape)==3 :
                            #        b,g,r = cv2.split(img_t)       # get b,g,r    
                            #        plt.imshow(cv2.merge([r,g,b]))
                            #    else:
                            #        plt.imshow(img_t)
                            #    plt.savefig(os.path.join(self.Testingdir_out,'kps_' + filename + '.png'))
                            #    plt.close()            
                   
                            goodMatchs=self.category.match_features_with_reduced_pos(test_kps[1],self.algorithm,self.accuracy)
                            m.log('good matches = '+ str(len(goodMatchs)),self.log_file)

                            for g in goodMatchs:
                                good_kps.append(np.float32(test_kps[0][g.queryIdx].pt)) #to be used in clustering 
            
                            good_kps=np.asarray(good_kps)
            
                            if self.localizer==m.LOCALIZER_DBSCAN:
                                m.log("Using DBScan to Localize Targets...")
                                cluster_centers,cluster_labels=m.cluster_features_dbscan(test_kps,goodMatchs,good_kps,self.epsilon,self.min_pts_per_cluster,bandwidth=self.category.bandwidth)
                                #cc_f=m.filter_features_clusters2(cluster_centers,cluster_labels,good_kps,self.category.bandwidth)
                            elif self.localizer==m.LOACLIZER_MEANSHIFT:
                                m.log("Using Mean Shift to Localize Targets...")
                                cluster_centers,cluster_labels=m.cluster_features_meanshift(test_kps,goodMatchs,good_kps,self.category.bandwidth)
                            #Create images of results and save it(self.graph -> 0: No Graphs, 1: create and save but no show, 2: create and save and show)
                            if self.graph:
                                m.plot_features(img,good_kps,1,os.path.join(self.Testingdir_out, 'good_kps_'+ testfile))
                            
                        cc_f=[]
                        cc_p=[]
                        hist_all=[]
                        self.algorithm.bowextractor.setVocabulary(self.category.vocab)

                        for c in cluster_centers:
                            #print(c[1]-band/2,c[1]+band/2,c[0]-band/2,c[0]+band/2)
                            patch=img[c[1]-band/2:c[1]+band/2,c[0]-band/2:c[0]+band/2]     #TODO may need enlargement

                            #patch_kps=self.algorithm.detector.detect(patch)
                            patch_kps,patch_descs=self.algorithm.descriptor.detectAndCompute(patch,None)
                            #m.plot_features2(patch,patch_kps,0)

                            hist=self.algorithm.bowextractor.compute(patch,patch_kps,patch_descs)
                            if hist is not None:
                                cc_f.extend(self.category.SVM.predict(hist))
                                cc_p.extend(self.category.SVM.predict_proba(hist))
                            else:
                                cc_f.extend([0])
                                hist=np.zeros(len(self.category.vocab), dtype=np.float32)
                            
                            hist_all.extend(hist) 
                            
                                
                            if self.graph>2 and patch_kps!=[]:
                                
                                p_t=cv2.drawKeypoints(patch,patch_kps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                if len(patch.shape)==3 :
                                    b,g,r = cv2.split(p_t)       # get b,g,r    
                                    plt.imshow(cv2.merge([r,g,b]))
                                else:
                                    plt.imshow(p_t)
                                #plt.savefig(os.path.join(self.Testingdir_out,'kps_' + filename + '.png'))
                                plt.show()                                                             
                            patch=None                  
                        #hist_all=np.asarray(hist_all)
                        #df=self.category.SVM.decision_function(hist_all)

                        detected_targets_for_file = m.pixelToLatLon(testFile_path,cluster_centers,cc_f)        
                        #Filter by score of the cluster         
                        #for k in range(len(cc_f)):
                        #    m.log(detected_targets_for_file[k] + ' : ' + cc_f[k],self.log_file,print_=1)                        
                        

                        TT=m.load_truth_table2(gtfile)

                        m.log('Detected Objects all:  ' + str(len(cluster_centers)),self.log_file)
                        m.log('Filtered Detected Objects:  ' + str(np.sum(cc_f)),self.log_file)
                        m.log('True Objects:  ' + str(len(TT)),self.log_file)

               
                        
 
                    
                        #Create images of results and save it(self.graph -> 0: No Graphs, 1: create and save but no show, 2: create and save and show)
                        if self.graph:
                            m.plot_features_and_cluster_centers(img,cluster_centers,None,cluster_labels,good_kps, self.graph,
                                                                os.path.join(self.Testingdir_out,'objects_'+ testfile))
                            m.plot_features_and_cluster_centers(img,cluster_centers,cc_f,cluster_labels,good_kps, self.graph,
                                                                os.path.join(self.Testingdir_out,'filtered_objects_'+ testfile))
                        img=None
                        ##Save detected targets as shape file
                        ##Disabled during test HAN Dataset
                        if self.geo:
                            self.save_results_as_shp(filename,detected_targets_for_file)
                        
                        #HanIncompatible
                        #Caculate stats of operation for this file
                        #self.results=[self.precision,self.recall,self.tpos,self.fpos,self.fneg,self.fecho]
                        if self.geo:
                            results_for_file=self.category.get_detection_stats(TT,detected_targets_for_file)
                        else:
                            results_for_file=self.category.get_detection_stats2(TT,detected_targets_for_file)

                        self.results = [self.results[i] + results_for_file[i] for i in range(6)]
            
                        m.log('Precision : '+ str(results_for_file[0]),self.log_file)
                        m.log('recall : '+ str(results_for_file[1]),self.log_file)
                        m.log('True Positives: '+ str(results_for_file[2]),self.log_file)
                        m.log('False Positives: '+ str(results_for_file[3]),self.log_file)
                        m.log('False Negatives:'+ str(results_for_file[4]),self.log_file)
                        m.log('False Echo:'+ str(results_for_file[5]),self.log_file)
            if self.results[2]+self.results[3] > 0 :
                self.results[0] = (self.results[2]/(self.results[2]+self.results[3]))
            else:
                self.results[0] = 0
            self.results[1] = self.results[2]/(self.results[2]+self.results[4])         
        
            m.log('Precision : '+ str(self.results[0]),self.log_file)
            m.log('recall : '+ str(self.results[1]),self.log_file)
            m.log('True Positives: '+ str(self.results[2]),self.log_file)
            m.log('False Positives: '+ str(self.results[3]),self.log_file)
            m.log('False Negatives:'+ str(self.results[4]),self.log_file)
            m.log('False Echo:'+ str(self.results[5]),self.log_file)
        
            self.endTime=datetime.now()#.strftime('%Y%m%d_%H%M%S.%f')
            self.timeDelta=self.endTime-self.startTime
        
            self.saveOperation(conn)

            conn.close()


    def saveOperation(self,conn):
        #conn = sqlite3.connect(m.db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        startTime=self.startTime.strftime('%Y%m%d_%H%M%S.%f')
        endTime=self.endTime.strftime('%Y%m%d_%H%M%S.%f')
        timedelta=self.timeDelta.total_seconds()
        #   .strftime('%H:%M:%S.%f')

        cur.execute("insert into OPS_OUT (start_time,end_time,epsilon,min_pts_per_cluster,accuracy,precision,recall,time,tpos,fpos,fneg,echo,algorithm,bandwidth,notes,category) values ( ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (startTime,endTime,self.epsilon,self.min_pts_per_cluster,self.accuracy,self.results[0],self.results[1],timedelta,self.results[2],self.results[3],self.results[4],self.results[5],self.algorithm.id,self.category.bandwidth,self.notes,self.category.id))
        conn.commit()

    def save_results_as_shp(self,filename,detected_targets):
        # set up the shapefile driver
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # create the data source
        data_source = driver.CreateDataSource(os.path.join(self.Testingdir_out, filename + "_results.shp"))

        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # create the layer
        layer = data_source.CreateLayer("detected_objects", srs, ogr.wkbPoint)

        field_name = ogr.FieldDefn("ID", ogr.OFTInteger)
        layer.CreateField(field_name)
        layer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))

        for i,r in zip(range(len(detected_targets)),detected_targets):
            # create the feature
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("ID", i)
            feature.SetField("Latitude", r[1])
            feature.SetField("Longitude", r[0])

            # create the WKT for the feature using Python string formatting
            wkt = "POINT(%f %f)" %  (float(r[1]) , float(r[0]))

            # Create the point from the Well Known Txt
            point = ogr.CreateGeometryFromWkt(wkt)

            # Set the feature geometry using the point
            feature.SetGeometry(point)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Destroy the feature to free resources
            feature.Destroy()

        # Destroy the data source to free resources
        data_source.Destroy()

        #### Get True positives,True Negatives, False Positive, Echo

class objfindTile():
    def __init__(self,tile):
        self.good_kps=[]

    def run(self):
        goodMatchs=[]
        img=cv2.imread(tile_path,cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        test_kps=self.algorithm.get_kps(img)
        goodMatchs=self.category.match_features_FLANN(test_kps[1],self.accuracy)
        for g in goodMatchs:
            self.good_kps.append(np.float32(test_kps[0][g.queryIdx].pt)) #to be used in  
        self.good_kps=np.asarray(good_kps)
        