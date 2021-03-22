## Extract Features
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr extract --binaries /data/benign_unpacked/

$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr extract --binaries /data/malicious_unpacked/
```

## Train Model
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr train --features /data/benign_unpacked_bndb_raw_features/ \
                              --output /data/model/
```

## Extract RoIs
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr roi --feature /data/malicious_unpacked_bndb_raw_features/ \
                            --bndb-func /data/malicious_unpacked_bndb_function/ \
                            --model /data/model/ \
                            --thresh 9.053894787328584e-08 \
                            --out-mse /data/malicious_unpacked_bndb_raw_features_mse/ \
                            --out-roi /data/roi/
```

## Cluster
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr cluster --bndb-func /data/malicious_unpacked_bndb_function/ \
                                --roi /data/roi/ \
                                --output /data/cluster/
```
