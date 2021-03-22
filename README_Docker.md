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
                --rm dr train --features /data/benign_unpacked_bndb_raw_feature/ \
                              --output /data/model/
```

## Extract RoIs
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr roi --feature /data/malicious_unpacked_bndb_raw_feature/ \
                            --bndb-func /data/malicious_unpacked_bndb_function/ \
                            --mse /data/malicious_unpacked_bndb_raw_feature_mse/ \
                            --thresh 9.053894787328584e-08 \
                            --output /data/roi/
```

## Cluster
```
$ docker run    --mount type=bind,source=/data/,target=/data/ \
                --rm dr cluster --bndb-func /data/malicious_unpacked_bndb_function/ \
                                --roi /data/roi/ \
                                --output /data/cluster/
```
