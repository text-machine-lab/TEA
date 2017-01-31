import os
from code.config import env_paths
import cPickle
from code.notes import TimeNote
import pprint
import glob

pp = pprint.PrettyPrinter(indent=4)
#gold='ABC19980114.1830.0611.tml'
#plain='ABC19980114.1830.0611.tml.TE3input'
#tmp_note = TimeNote.TimeNote('training_data/'+plain, 'training_data/'+gold)
#pp.pprint(tmp_note.get_tlinked_entities())
import re

file_list = glob.glob('/home/ymeng/projects/sandbox/training_set/*.tml')

for fname in file_list:
    f = open(fname)
    out_name = '/home/ymeng/projects/sandbox/training_set_augmented/'+fname.split('/')[-1]
    with open(out_name, 'w') as fout:
        for line in f:
            fout.write(line)
            if '<TLINK' not in line: 
                continue
            else:
                origin = line
                lid = re.search('lid=\"(l\d+)\"', origin).group(1)
                line = re.sub('lid=\"(l\d+)\"', 'lid=\"'+lid+'9999\"', origin)

                if 'BEFORE' in line: 
                    line = line.replace('BEFORE', 'AFTER')
                elif 'AFTER' in line:
                    line = line.replace('AFTER', 'BEFORE') 
    
                if 'INCLUDES' in line: 
                    line = line.replace('INCLUDES', 'IS_INCLUDED')
                elif 'IS_INCLUDED' in line: 
                    line = line.replace('IS_INCLUDED', 'INCLUDES')
                
                if 'BEGINS' in line: 
                    line = line.replace('BEGINS', 'BEGUN_BY')
                elif 'BEGUN_BY' in line: 
                    line = line.replace('BEGUN_BY', 'BEGINS')
                
                if 'ENDS' in line: 
                    line = line.replace('ENDS', 'ENDED_BY')
                elif 'ENDED_BY' in line: 
                    line = line.replace('ENDED_BY', 'ENDS')
    
                if 'relatedToTime' in line:
                    line = line.replace('relatedToTime', 'time-ID')
                    line = line.replace('timeID', 'relatedToTime')
                    line = line.replace('eventInstanceID', 'relatedToEventInstance')
                    line = line.replace('time-ID', 'timeID')
                elif 'relatedToEventInstance' in line:
                    line = line.replace('relatedToEventInstance', 'event-Instance_ID')
                    line = line.replace('timeID', 'relatedToTime')          
                    line = line.replace('eventInstanceID', 'relatedToEventInstance')
                    line = line.replace('event-Instance_ID', 'eventInstanceID')
                # pp.pprint(line)
                fout.write(line) 
    
    f.close()           
