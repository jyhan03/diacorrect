#!/usr/bin/env bash
#

score_area=
collar=0

. ./utils/parse_options.sh
ref_rttm_path=$1
hyp_rttm_path=$2

./utils/md-eval-22.pl $score_area -c $collar -afc -r $ref_rttm_path -s $hyp_rttm_path > temp/temp.info
grep SCORED temp/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > temp/SCORED.list
grep MISSED temp/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > temp/MISSED.list
grep FALARM temp/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > temp/FALARM.list
grep "SPEAKER ERROR" temp/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > temp/SPEAKER.list
grep OVERALL temp/temp.info | cut -d "=" -f 4 | cut -d ")" -f 1 > temp/session.list
sed -i '$d' temp/session.list
echo "ALL" >> temp/session.list
for l in `cat temp/session.list`;do
    grep $l $ref_rttm_path | awk '{print $8}' | sort | uniq | wc -l
done > temp/oracle_spknum.list

for l in `cat temp/session.list`;do
    grep $l $hyp_rttm_path | awk '{print $8}' | sort | uniq | wc -l
done > temp/diarized_spknum.list

paste -d " " temp/session.list temp/SCORED.list temp/MISSED.list \
             temp/FALARM.list temp/SPEAKER.list temp/oracle_spknum.list \
             temp/diarized_spknum.list > temp/temp.details

awk '{printf "%s %.2f %.2f %.2f %.2f %d %d\n",$1,$4/$2*100,$3/$2*100,$5/$2*100,($3+$4+$5)/$2*100,$6,$7}' temp/temp.details > temp/temp.info1
echo "session FA MISS SPKERR DER ORACLE_SPKNUM DIARIZED_SPKNUM" > temp/temp.details
grep -v "ALL" temp/temp.info1 | sort -n -k 5 >> temp/temp.details
grep "ALL" temp/temp.info1 >> temp/temp.details

# echo main information
echo "session FA MISS SPKERR DER ORACLE_SPKNUM DIARIZED_SPKNUM" > temp/temp.main
grep "ALL" temp/temp.info1 >> temp/temp.main
column -t temp/temp.main

#column -t temp/temp.details
