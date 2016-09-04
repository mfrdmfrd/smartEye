
# coding: utf-8

import os
import mylibrary as m
import pickle


# In[2]:

#importlib.reload(lib2)
#importlib.reload(m)


# In[3]:




# In[4]:

data_root='.\\Data\\'
dbname="train.db"
m.initialize_parameteres(data_root_=data_root,dbname=dbname)

k=1000
n=256
m.retrain=0       #2:rebuild classifier, 1:rebuild vocabulary, 0:Don't rebuild
m.gray=0
m.bow_encoding=0                #0: Hard, 1: Soft, 2: LLC
saveCategories=0

# In[5]:

sift=m.Algorithm('SIFT',k=k,n=n)
surf=m.Algorithm('SURF',k=k,n=n)


# In[5]:
if m.retrain==0:
    ##################Load pickle db###################
    #cat_sift=m.loadCategories_pickle(sift)
    #cat_surf=m.loadCategories_pickle(surf)
    ##################Load pickle file#################
    #cat_surf=m.load_categories_from_file('.//images//surf')
    cat_sift=m.load_categories_from_file('.//images//sift')
    #cat_sift[0]=m.load_category_from_file('airplane','.//images//sift')
elif m.retrain==1:
    ###################Train##################################
    cat_sift=m.loadCategories(m.retrain)
    cat_surf=m.loadCategories(m.retrain)
    for algorithm,cats in zip([sift,surf],[cat_sift,cat_surf]):
        for c in cats:
           c.train_classifier(algorithm)
elif m.retrain==2:
    #################Rebuild classifier########################
    #for ,cats in zip([sift,surf],[cat_sift,cat_surf]):
    for c in cat_sift:
        c.X_train=[]
        c.train_classifier(algorithm)
#categories_s=[cat_sift,cat_surf]


# In[10]:

times=[]
algorithm=sift
categories=cat_sift

m.graph=0

m.localizer=1
m.window_size_factor=1
m.window_overlapping_factor=0
m.epsilon=0.05
m.min_pts_per_cluster=1
m.reduced=0.2
m.accuracy=0.85

#for accuracy in np.arange(0.80,1.01,0.01):
#    m.accuracy=accuracy
m.test(categories,algorithm)



#cat_sift[0].saveTrainData2(sift)





# In[9]:

#########################Save###############################
if saveCategories:
    m.save_all_categories(cat_sift,'.//images//sift')           
    #m.save_all_categories(cat_surf,'.//images//surf')           
    #for algorithm,cats in zip([sift,surf],[cat_sift,cat_surf]):
    #    for c in cats:
    #        c.saveTrainData2(algorithm)





















# surf.pca_n=256
# sift.pca_n=256


# In[ ]:



# In[13]:

#for algorithm,cats in zip([sift,surf],categories_s):
#    for C in cats:
#         print(C.PCA)
#         print(len(C.training_data_all))
#         C.saveTrainData(algorithm)


# roc_aucs=[]
# pr_aucs=[]
# wsf=[]
# sof=[]
# for window_size_factor in np.arange(1.0,2.0,.2):
#     for window_overlapping_factor in np.arange(0.0,.5,0.1):
# for algorithm,categories in zip([sift,surf],categories_s):  
#     t=

#for accuracy in [0.7,0.75,0.8,0.85,0.9,0.95]:
#    for localizer in [1,2]:
#     times.append(t)
    
# roc_aucs.append(roc_auc)
# pr_aucs.append(pr_auc)
# wsf.append(window_size_factor)
# sof.append(window_overlapping_factor)


# In[97]:
#tt_sift=[]
#tt_surf=[]
#for t in times[0]:
#    tt_sift.append(t.total_seconds())
#for t in times[1]:
#    tt_surf.append(t.total_seconds())
#plt.clf()
#plt.plot(range(len(tt_sift)),tt_sift, label='SIFT Average={0:0.2f}'.format(np.average(tt_sift)))
#plt.plot(range(len(tt_surf)),tt_surf, label='SURF Average={0:0.2f}'.format(np.average(tt_surf)))
#plt.xlabel('')
#plt.ylabel('Time (secs)')
#plt.title('Time consumed per image patch')
#plt.legend(loc="upper right")
#plt.savefig(os.path.join('.//images','time_consumed.png'))
#plt.show()


# In[182]:

#ns=[128,256,512]
#for n in ns:
#for category in cat_surf:
##         category.X_train=np.array(category.SVMTrainData)
##         category.y_train=np.array(category.SVMTrainLabel)
##         category.build_pca(sift)
##         category.train_classifier()
#    category.saveTrainData(surf)


