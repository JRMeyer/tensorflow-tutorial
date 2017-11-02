#!/bin/bash

# Joshua Meyer (2017)

# Input: dataset used in Kaggle <https://www.kaggle.com/uciml/sms-spam-collection-dataset>, originally from <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>
#
# This script just splits above dataset into format that
# works for input script, namely:
#
#  data/
#     ham/
#       hamText1.txt
#       hamText2.txt
#       hamText3.txt
#
#     spam/
#       spamText1.txt
#       spamText2.txt
#       spamText3.txt


SMSSpamCollection=$1

if [ ! -f $SMSSpamCollection ]; then
   echo "$0: Input dataset not found!"
fi

mkdir -p data/ham data/spam
   
iter=0;
while read line;
do line=( $line );
   label=${line[0]};
   if [ "$label" == "ham" ]; then
       sms="${line[@]:3}";
   else
       sms="${line[@]:4}";
   fi;

   # don't know why but I'm getting empty sms files,
   # just ignore them (< 100 out of thousands)
   if [ "$sms" == "" ]; then
      echo "empty SMS:/ Skipping..."
   else
       echo $sms > data/${label}/${iter}.txt;
       ((iter++));
   fi
        
done<$SMSSpamCollection
