BEGIN TRANSACTION;
CREATE TABLE `trainingDataArchive` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`descriptor`	array,
	`category`	INTEGER,
	`y`	INTEGER,
	`x`	TEXT,
	`size`	REAL,
	`angel`	INTEGER,
	`octave`	INTEGER,
	`responce`	REAL,
	`algorithm`	INTEGER,
	`positive`	INTEGER,
	`image`	INTEGER
);
CREATE TABLE `trainingData` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`descriptor`	array,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`positive`	INTEGER
);
CREATE TABLE `ops_out` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`op_id_txt`	TEXT,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`op_id_txt`	TEXT,
	`start_time`	TEXT,
	`end_time`	TEXT,
	`time`	TEXT,
	`accuracy`	INTEGER,
	`precision`	REAL,
	`recall`	REAL,
	`tpos`	TEXT,
	`fpos`	INTEGER,
	`fneg`	INTEGER,
	`echo`	INTEGER,
	`bandwidth`	REAL,
	`Notes`	TEXT,
	`min_pts_per_cluster`	NUMERIC,
	`epsilon`	REAL,
	`localizer`	INTEGER,
	`wsf`	REAL,
	`wof`	REAL,
	`reduced_k`	REAL,
	`y_test`	BLOB,
	`y_score`	BLOB,
	`soft_encoding`	INTEGER
);CREATE TABLE `category` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`NAME`	TEXT NOT NULL,
	`bandwidth`	REAL,
	`SVM`	ARRAY,
	`PCA`	ARRAY,
	`algorithm`	INTEGER
);
CREATE TABLE `algorithm` (
	`id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`name`	INTEGER,
	`bow`	INTEGER,
	`vocab_k`	REAL,
	`Full_train_data`	INTEGER,
	`dataset`	INTEGER,
	`localizer`	INTEGER,
	`wndows_factor`	NUMERIC,
	`windows_overlap_factor`	REAL,
	`min_pts`	INTEGER,
	`epsilon`	INTEGER,
	`reduced_traindata_k`	REAL,
	`svm`	array
);
CREATE TABLE `vocabData` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`descriptor`	array,
	`category`	INTEGER,
	`algorithm`	INTEGER
);
CREATE TABLE `classifier` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`SVM`	BLOB,
	`PCA`	BLOB,
	`traindata_pos`	BLOB,
	`traindata_neg`	BLOB,
	`vocab`	BLOB,
	`notes`	TEXT,
	`enabled`	INTEGER DEFAULT 1
);
CREATE TABLE `category_pickle` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`pickle`	ARRAY,
	`enabled`	INTEGER DEFAULT 1,
	`notes`	TEXT,
	`pca`	REAL,
	`coding`	INTEGER
);
CREATE TABLE `image` (
	`id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`name`	INTEGER,
	`path`	INTEGER,
	`positive`	INTEGER,
	`combo`	INTEGER
);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (1,'airplane',0,1);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (2,'airport',0,1);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (3,'car',0,1);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (1,'airplane',0,2);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (2,'airport',0,2);
INSERT INTO `category` (ID,NAME,bandwidth,algorithm) VALUES (3,'car',0,2);
INSERT INTO `algorithm` (id,name) VALUES (1,'SIFT');
INSERT INTO `algorithm` (id,name) VALUES (2,'SURF');
COMMIT;
