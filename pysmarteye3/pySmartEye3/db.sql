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
	`positive`	INTEGER
);
CREATE TABLE "trainingData" (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`descriptor`	array,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`positive`	INTEGER,
	`clusterCenter`	INTEGER
);
CREATE TABLE `ops_out` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`category`	INTEGER,
	`algorithm`	INTEGER,
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
);
CREATE TABLE "category" (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`NAME`	TEXT NOT NULL,
	`bandwidth`	REAL,
	`SVM` ARRAY,
	`PCA` ARRAY);
CREATE TABLE `algorithm` (
	`id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
	`name`	INTEGER
);
CREATE TABLE `vocabData` (
	`ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	`descriptor`	array,
	`category`	INTEGER,
	`algorithm`	INTEGER,
	`positive`	INTEGER,
	`clusterCenter`	INTEGER
);
INSERT INTO `category` (ID,NAME,bandwidth) VALUES (1,'airplane',0);
INSERT INTO `category` (ID,NAME,bandwidth) VALUES (2,'airport',0);
INSERT INTO `category` (ID,NAME,bandwidth) VALUES (3,'car',0);
INSERT INTO `algorithm` (id,name) VALUES (1,'SIFT');
INSERT INTO `algorithm` (id,name) VALUES (2,'SURF');
COMMIT;
