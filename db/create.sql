CREATE TABLE dr(
    unique_ID char(64) UNIQUE NOT NULL PRIMARY KEY,
    hash char(64),
    family varchar(128) DEFAULT 'unknown',
    label varchar(128) DEFAULT 'unlabeled',
    func_addr varchar(10),
    cid int DEFAULT -1,
    manually_labeled boolean DEFAULT False,
    labeled_by_cluster boolean DEFAULT False
);
