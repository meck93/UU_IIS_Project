# Network Parameter Tuning & Evaluation
This file contains the various parameters that I have experiemented with so far and the tensorboard log-files for the best runs.

Model: Test1
- new scoring
- dropout: 0.0
- lrschedule: decrease factor 0.5 every 4 epochs if no decrease in mean_euclid_dist

Model: Test2
- new scoring
- dropout: 0.1
- lrschedule: decrease factor 0.5 every 4 epochs if no decrease in mean_euclid_dist

What have I done so far? 
- experimented with batch size
- - 128 and 64 are too large and the val_mean_euclid_dist never gets manages to get lower than 0.15
- - 32 seems to be optimal
- Dropout
- - what is the optimal value? 0.0 or 0.1?
- Best Val Result
- - 0.063 (Dropout 0.0)
- - 0.0713 (Dropout 0.1)
