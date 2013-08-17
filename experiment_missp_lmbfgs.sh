#!bin/bash
FILE_MATCH=pos.txt
FILE_MISMATCH=neg.txt
FILE_FEATURES=feats.txt 
FILE_LOG=log.txt 
FILE_PARAMS=params.txt 
FILE_MATCH_VAL=pos.txt 
FILE_MISMATCH_VAL=neg.txt

C=0
for L in 0.1 #1.0 10.0
do
    python train_lmbfgs_script.py $FILE_MATCH $FILE_MISMATCH feats.txt $FILE_LOG $FILE_PARAMS $FILE_MATCH_VAL $FILE_MISMATCH_VAL $L 
    #wait
    #C=$[C+1]
done

