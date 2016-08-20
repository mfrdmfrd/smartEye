from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import threading
import mylibrary as m
import ObjFind as o


#results = []
#accuracys=np.arange(0.71,0.74,0.01)
#epsilon=np.arange(0.05,0.05,0.05)
#min_pts=np.arange(1,10)

#for pts in range(10):
#    p.initialize_parameteres(min_pts_per_cluster_=pts)
#    prec,recall=o.objfind()
#    results.append([prec,recall])



###Initialize
m.initialize_parameteres(data_root_='E:\\Master\\Data\\',dbname="train.db")
Alg=m.Algorithm('SIFT',bow=True,k=100)


#ops = []
#n=1
#    op = o.objfind(accuracy_=a,i=n)
#    ops += [op]
#    op.start()
#    n+=1

#for x in ops: 
#    x.join()
#    print('Thread ' +  x.op_id + ' finished')
#        

categories=m.loadCategories()
category=categories[0]
for category in categories:
    #for acc in np.arange(0.1,.95,0.05):
    #    for min_pts in range(1,5):
    #        op = o.objfind(algorithm=Alg,category=category,min_pts_per_cluster_=min_pts,
    #                    accuracy_=acc,graph=2,geo=0,testpath=category.cat_testdata_path,dbscan=0)
    #       op.run()
    m.log ('Start Training ' + category.name + ' With algorithm ' + Alg.detector_name)
    category.train3(Alg,0)     # for not ask to retrain
    if len(category.vocab)>0:
        notes='Testing' 
        op = o.objfind(algorithm=Alg,
                       category=category,
                       min_pts_per_cluster_=3,
                       accuracy_=.8,
                       graph=0,
                       geo=0,
                       testpath=category.cat_testdata_path,
                       dbscan=1,
                       notes=notes,
                       epsilon_=.15,
                       localizer=m.LOCALIZER_WINDOW,
                       window_size_factor=1,
                       window_overlapping_factor=.2)
        op.run_bow()




###Test
#for category in categories:
#    for min_pts in range(1,25):
#        for acc in np.arange(0.0,1.0,0.1):
#            for bw in range(10,200,10):
#                for cluster_db in [0,1]:
#                    for eps in np.arange(0.1,0.3,0.1):
#                        op = o.objfind(algorithm=Alg,category=category,min_pts_per_cluster_=min_pts,
#                                       accuracy_=acc,bandwidth=bw,dbscan=cluster_db,epsilon_=eps,graph=2)
#                        op.run()
        











#results.append([op.precision,op.recall])
        #category.results=[]


##results =[[0, 0], [0, 0], [0, 0], [0, 0], [1.0, 0.25], [1.0, 0.25], [0.8, 1.0], [0.4, 1.0], [0.2222222222222222, 1.0], [0.19047619047619047, 1.0]]

##results=sorted(results, key=lambda x: x[0])

#r=list(zip(*results))
#n=len(results)

#plt.plot(epsilon,r[0],color='red',label='Precision')
#plt.plot(epsilon,r[1],color='blue',label='Recall')

#plt.ylim(0,101)
#plt.xlim(0,0.6)
##red_patch = mpatches.Patch(color='red', label='Precision')
##blue_patch = mpatches.Patch(color='blue', label='Recall')

##plt.legend(handles=[red_patch,blue_patch])
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
##plt.title('Precision and Recall with Accuracy graph')
#plt.ylabel('Precision/Recall')
#plt.xlabel('Epsilon')
##plt.savefig(os.path.join(p.Testingdir_out,'p-r_graph')+'.png')
##plt.show()


#plt.savefig(os.path.join(p.Testingdir_out,'p-r_graph')+'.png')
#plt.show()



#plt.plot(r[1],r[0])
#plt.title('Precision-recall Graph for Accuracy = {}, min points = {}'.format(p.accuracy, p.min_pts_per_cluster))
#plt.ylabel('Recall')
#plt.xlabel('Precision')
#plt.ylim(0,1.1)
#plt.xlim(0,1.1)

#plt.savefig(os.path.join(p.Testingdir_out,'pr_graph')+'.png')
#plt.show()
