import sqlite3
import numpy as np
import io
import cv2
import mylibrary as m

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


conn = sqlite3.connect('train.db', detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
print ("Opened database successfully")

try:
    conn.execute('''SELECT * FROM category;''')
except:
    conn.execute('''CREATE TABLE category
           (ID INT PRIMARY KEY     NOT NULL,
           NAME           TEXT    NOT NULL);''')

    conn.execute('''CREATE TABLE trainingData
           (ID INT PRIMARY KEY     NOT NULL,
           descriptor   array);''')
        
def saveTrainData(trainData,category,algorithm):
    
    kps=trainData[0]
    descs=trainData
    cur=conn.cursor()
    cur.execute("select ID from category where name = ?",(category,))
    exist=cur.fetchone()
    if exist:
        c_id=exist[0]
    cur.execute("select id from algorithm where name = ?",(algorithm,))
    exist=cur.fetchone()
    if exist:
        alg_id=exist[0]
    conn.execute("delete from trainingData where category=?",(c_id,))

    for des in descs:
        #conn.execute("insert into trainingData (x,y,size,angel,octave,responce,descriptor,category,algorithm)values(?,?,?,?,?,?,?,?,?);",
        #             (kps[i].pt[0], kps[i].pt[1], kps[i].size, kps[i].angle, kps[i].octave, kps[i].response, descs[i], c_id, alg_id))
        conn.execute("insert into trainingData (descriptor,category,algorithm)values(?,?,?)",
                     (des, c_id, alg_id))
    conn.commit()

def loadTrainData(catgeory,algorithm):
    cur = conn.cursor()
    kps=[]
    descs=[]
    #cur.execute("SELECT x,y,size,angel,octave,responce,descriptor from (trainingData t join category c on t.category=c.id) join algorithm a on t.algorithm=a.id where c.name = ? and a.name=?" ,(catgeory , algorithm) )
    cur.execute("SELECT descriptor from (trainingData t join category c on t.category=c.id) join algorithm a on t.algorithm=a.id where c.name = ? and a.name=?" ,(catgeory , algorithm) )
    for r in cur:
        #kp=cv2.KeyPoint(np.float32(r[1]),np.float32(r[0]),np.float32(r[2]))
        #kp.angle=r[3]
        #kp.octave=r[4]
        #kp.response=r[5]
        #kps.append(kp)
        desc=r[0]
        descs.append(desc)

    return [kps,descs]

def saveCategory(cat_name,bw):
    cur = conn.cursor()
    cur.execute("select id from category where name = ?",(cat_name,))
    exist = cur.fetchone()
    if exist:
        c_id=exist[0]
        cur.execute("update category set name = ? , bandwidth = ? where id = ?", (c_id,cat_name, bw ))
    else:
        cur.execute("insert into category (name,bandwidth) values (?,?)" , (cat_name,bw))


    conn.commit()

def loadCategories():
    cats=[]
    cur = conn.cursor()
    cur.execute("select id,name,bandwidth from category ")
    for r in cur:
        c=m.Category(r[0],r[1],r[2])
        cats.append(c)
    return cats