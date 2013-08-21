#!bin/bash
FILE_MATCH=pos.txt        # File containing positive training examples
FILE_MISMATCH=neg.txt     # File containing negative training examples
FILE_FEATURES=feats.txt   # Features to include
FILE_LOG=log.txt          # Log file to write training progress 
FILE_PARAMS=params.txt    # Model parameters
FILE_MATCH_VAL=pos.txt    # Find accuracy at each training iteration by 
FILE_MISMATCH_VAL=neg.txt #     evaluating on training examples (would 
                          #     typically use different examples here)

# Train on examples
./learning -lG -iF $FILE_FEATURES \
    -iM $FILE_MATCH -iN $FILE_MISMATCH \
    -iV $FILE_MATCH -iW $FILE_MISMATCH \
    -oP $FILE_PARAMS \
    -oL $FILE_LOG\
    -lL 0.1  # l2 regularization

# Test examples
./learning -iF $FILE_FEATURES \
    -iV $FILE_MATCH -iW $FILE_MISMATCH \
    -oP $FILE_PARAMS 

# Score examples
./learning -S -iF $FILE_FEATURES \
    -iM $FILE_MATCH -iN $FILE_MISMATCH \
    -oP $FILE_PARAMS 
