/* Malware functions */
CREATE TABLE functions(
    id INT GENERATED ALWAYS AS IDENTITY,
    /* SHA256 hash of sample */
    hash CHAR(64) NOT NULL,
    /* Family label */
    family VARCHAR(128) DEFAULT 'unknown',
    /* Function address */
    func_addr VARCHAR(18) NOT NULL,
    /* Cluster that function belongs to */
    cid INT DEFAULT -2,
    /* DeepReflect score of function */
    score FLOAT DEFAULT -1,
    /* Define unique entries */
    UNIQUE (hash,family,func_addr),
    /* Define keys */
    PRIMARY KEY(id)
);

/* Function labels */
CREATE TABLE labels(
    id INT GENERATED ALWAYS AS IDENTITY,
    /* Label's plain name */
    name VARCHAR(128) NOT NULL,
    /* Define unique entries */
    UNIQUE (name),
    /* Define keys */
    PRIMARY KEY(id)
);

/* Analysts who labeled functions */
CREATE TABLE analysts(
    id INT GENERATED ALWAYS AS IDENTITY,
    /* Analyst's username */
    username VARCHAR(64) NOT NULL,
    /* Analyst's email */
    email VARCHAR(64) NOT NULL,
    /* Analyst's plain name */
    name VARCHAR(64) NOT NULL,
    /* Define unique entries */
    UNIQUE (username),
    /* Define keys */
    PRIMARY KEY(id)
);

/* Function reviews */
CREATE TABLE reviews(
    id INT GENERATED ALWAYS AS IDENTITY,
    /* Function id */
    function_id INT NOT NULL,
    /* Label id */
    label_id INT NOT NULL,
    /* Analyst id */
    analyst_id INT NOT NULL,
    /* Timestamp of entry */
    ts_start TIMESTAMP NOT NULL,                /* when user began looking at this function */
    ts_end TIMESTAMP DEFAULT CURRENT_TIMESTAMP, /* when user made their final decision about this function */
    /* Define keys */
    PRIMARY KEY(id),
    CONSTRAINT fk_functions
        FOREIGN KEY(function_id)
            REFERENCES functions(id)
            ON DELETE CASCADE,
    CONSTRAINT fk_labels
        FOREIGN KEY(label_id)
            REFERENCES labels(id)
            ON DELETE CASCADE,
    CONSTRAINT fk_analysts
        FOREIGN KEY(analyst_id)
        REFERENCES analysts(id)
        ON DELETE SET NULL
);

