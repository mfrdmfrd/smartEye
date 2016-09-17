import numpy as np
import matplotlib.pyplot as plt
import mylibrary as m
import sqlite3
from scipy.integrate import trapz, simps
from sklearn import metrics
import re
from  itertools import chain

m.initialize_parameteres()

csv_file='E:\Master\Paper\img\g2_1.txt'

algorithms=['DBM','LLC','BOW','pHOG','Thesis']

with open(csv_file, 'r') as csvfile:
    i=0
    x=[]
    y=[]
    # get number of columns
    lines=csvfile.readlines()
    while i < len(lines):
        if 'Line' in lines[i]:
            x_=[]
            y_=[]
            i+=1
            
            while i < len(lines) and lines[i] != '\n':
                temp=lines[i].replace("\"","").rstrip("\n").split(",")

                #temp = re.split("[, \!?:]+", lines[i])
                #temp = [word.strip() for word in lines[i]]
                #temp = lines[i].split(",")
                x_.append(float(temp[0]))
                y_.append(float(temp[1]))
                i+=1            
            x.append(x_)
            y.append(y_)
        i+=1

conn = sqlite3.connect(m.db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
cur.execute("SELECT recall,precision  from ops_out where id < 1385 and id > 1341")
x_=[]
y_=[]
for r in cur:
    x_.append(float(r[0]))
    y_.append(float(r[1]))
x.append(x_)
y.append(y_)    



               
color = 'cornflowerblue'
linestyles = ['-', '--', '-.', ':','-']
auc=[]  

plt.figure()
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precesion-Recall Curve on the ISPRS dataset')

for a,b,g,ls,i in zip(x,y,algorithms,linestyles,range(len(algorithms))):
    auc.append(0.0) #metrics.auc(a,b,True)
    #print(b)
    ll= g + ', AP=' + str(auc[i])
    p=plt.plot (a,b,label=ll,linestyle=ls,  linewidth=2)


plt.legend()
plt.show()                       
