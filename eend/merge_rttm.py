import os
import sys

rttm_path = sys.argv[1]

rttm_io = open(rttm_path)
merge_segment = 0.001
rttm = {}

for l in rttm_io:
    l = l.split()
    session, start, durance, spk = l[1], l[3], l[4], l[7]
    if session not in rttm.keys():
        rttm[session] = {}
    if spk not in rttm[session].keys():
        rttm[session][spk] = []
    rttm[session][spk].append([float(start), float(start)+float(durance)])
for session in rttm.keys():
    for spk in rttm[session].keys():
        rttm[session][spk] = sorted(rttm[session][spk], key = lambda x:x[0])

total_len = 0

for session in rttm.keys():
    for spk in rttm[session].keys():
        start, end = rttm[session][spk][0]
        for utt in rttm[session][spk][1:]:
            if utt[0] - end <= merge_segment:
                end = utt[1]
            else:
                #if end-start >= discard:
                print("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>".format(session, start, end-start, spk))
                #else:
                #    print("Discard SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>".format(session, start, end-start, spk))
                start, end = utt
        #if end-start >= discard:
        print("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>".format(session, start, end-start, spk))
        #else:
        #    print("Discard SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>".format(session, start, end-start, spk))
