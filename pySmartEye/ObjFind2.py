# coding: utf-8

import numpy as np
import cv2
import os
import matplotlib.cm as cm
import mylibrary as m
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
import parameters as p
import db

tiles=[]
test_data=[]



for file in os.listdir(p.Testingdir_in):    #ARG1
    if file.endswith(".tif"):
        tiles.append(file)

categories = db.loadCategories()
category=categories[0]        #ARG2
algorithm=m.Algorithm('SIFT')           #ARG3

category.train(algorithm)

m.log ('Start Testing' + category.name + ' With algorithm ' + algorithm.detector_name)

for tile in tiles:
    m.log('Start Testing '+ tile)
    
    tile_path=os.path.join(p.Testingdir_in, tile)
    

    filename, file_extension = os.path.splitext(tile)
        
        
    testing_out_path=os.path.join(p.Testingdir_out, category.name)
    
    testing_out_filepath=os.path.join(testing_out_path,tile)

    

    goods=[]
    good_kps=[]

    img=cv2.imread(tile_path,cv2.IMREAD_GRAYSCALE if p.gray else cv2.IMREAD_COLOR)
    test_kps=algorithm.get_kps(img)
    img_t=cv2.drawKeypoints(img,test_kps[0],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_t)
    plt.savefig(os.path.join(p.Testingdir_out,tile))
    plt.close()            
    #for t_des in train_des:
    goods=category.match_features_FLANN(test_kps[1],0.7)
    m.log('good matches = '+ str(len(goods)))

    for g in goods:
        good_kps.append(np.float32(test_kps[0][g.queryIdx].pt)) #to be used in clustering 
            
    good_kps=np.asarray(good_kps)
        
    #cluster_centers,cluster_labels=m.cluster_features(test_kps,goods,good_kps,category.get_bandwidth(tile_path))
    m.plot_features(img,good_kps)
    
    cluster_centers,cluster_labels=m.cluster_features_dbscan(test_kps,goods,good_kps,.2,3)
        #m.save_results(tile,Category,detector,descriptor,'mean_shift',cluster_centers,cluster_labels)

    m.log('Detected Objects all:  ' + str(len(cluster_centers)))
    cc_f=m.filter_features_clusters(cluster_centers,cluster_labels,3)
    m.log('Detected Objects filterd :  ' + str(len(cc_f)))
    #cc_f2=m.filterByHistogram(category,cc_f,img,tile_path)
    #m.log('Detected Objects filterd 2:  ' + str(len(cc_f2)))

    
    category.detected_coords = m.pixelToLatLon(tile_path,cc_f)        
    
    m.log('Detected Objects:  ')
    m.log(category.detected_coords)
        
    category.results.extend(category.detected_coords)
            
    #m.plot_features_cluster_centers(img,cc_f,2,category.name)
    m.plot_features_and_cluster_centers(img,cc_f,cluster_labels,good_kps, 1,category.name)

category.save_results(p.Testingdir_out)
            
TT=m.load_truth_table(os.path.join(p.Testingdir_in, category.name+'_tt.csv'))

precision,recall,tpos,fpos,fneg,fecho=category.get_detection_stats(TT,category.results)
            
m.log('Precision : '+ str(precision))
m.log('recall : '+ str(recall))
m.log('True Positives: '+ str(tpos))
m.log('False Positives: '+ str(fpos))
m.log('False Negatives:'+ str(fneg))
m.log('False Echo:'+ str(fecho))
            
#precisions=[]
#recalls=[]
#for f in range(30):
#    cc_f=m.filter_features_clusters(cluster_centers,cluster_labels,f)
#    precision,recall,tpos,fpos,fneg,fecho=m.get_detection_stats(TT,cc_f,bandwidth)
#    precisions.append(precision)
#    recalls.append(recall)

#linestyles = (['-', '--', ':'])
#color = ('bgrcmyk')
#markers = (["x","s","d"])


#    x=range(30)
#    plt.plot(x,np.asarray(precisions),linestyle=linestyles[0], marker=markers[0], color=color[0], markersize=4)
#    plt.plot(x,np.asarray(recalls),linestyle=linestyles[1], marker=markers[1], color=color[1], markersize=4)
#    plt.show()

#    cc_f=m.filter_features_clusters(cluster_centers,cluster_labels,recalls.index(max(recalls)))
#    precision,recall,tpos,fpos,fneg,fecho=m.get_detection_stats(TT,cc_f,bandwidth)
#    print('Precision : ',precision)
#    print('recall : ', recall)
#    print('True Positives: ', tpos)
#    print('False Positives: ', fpos)
#    print('False Negatives:', fneg)
#    print('False Echo:', fecho)

#m.plot_features_cluster_centers(img,cc_f,1,testing_out_filepath)
            
            


