#!bin/bash
FILE_MATCH=pos.txt
FILE_MISMATCH=neg.txt
FILE_FEATURES=feats.txt 
FILE_LOG=log.txt 
FILE_PARAMS=params.txt 
FILE_MATCH_VAL=pos.txt 
FILE_MISMATCH_VAL=neg.txt

rm $FILE_PARAMS

#Train on examples
./learning -lG -iF $FILE_FEATURES \
    -iM $FILE_MATCH -iN $FILE_MISMATCH \
    -iV $FILE_MATCH -iW $FILE_MISMATCH \
    -oP $FILE_PARAMS \
    -oL $FILE_LOG\
    -lL 0.1 #l2 regularization

#Test examples
./learning -iF $FILE_FEATURES \
    -iV $FILE_MATCH -iW $FILE_MISMATCH \
    -oP $FILE_PARAMS 

#Score examples
./learning -S -iF $FILE_FEATURES \
    -iM $FILE_MATCH -iN $FILE_MISMATCH \
    -oP $FILE_PARAMS 
