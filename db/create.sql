CREATE TABLE dr(
    /* Hash of sample hash, family, address */
    unique_ID char(64) UNIQUE NOT NULL PRIMARY KEY,
    /* SHA256 hash of sample */
    hash char(64),
    /* family label */
    family varchar(128) DEFAULT 'unknown',
    /* function label */
    label varchar(128) DEFAULT 'unlabeled',
    /* function address */
    func_addr varchar(18),
    /* cluster that function belongs to */
    cid int DEFAULT -2,
    /* if function has been manually labeled by analyst (prevents cluster associated labels) */
    manually_labeled boolean DEFAULT False,
    /* DeepReflect score of function */
    score float DEFAULT -1
);
