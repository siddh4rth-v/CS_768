cd graph-hashing/hash-gnn/hash-gnn
./hashgnn \
    -network ../data/googleplus/gooleplus.adjlist.0.8 \
    -feature ../data/googleplus/features.txt \
    -hashdim 200 \
    -iteration 3 \
    -embedding ../../../official_emb.txt \
    -time ../../../official_time.txt
cd ../../..