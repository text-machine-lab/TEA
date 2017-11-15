from code.notes.utilities.timeml_utilities import get_doctime_timex
import re

class TimeRefNetwork(object):
    def __init__(self, note):
        self.YEAR_END = 364.0/365
        self.MONTH_END = 29.0/365
        self.timex_elements = {}
        self.timex_id_pairs = []
        self.t0_value = None
        self.t0_real = None
        self.note = note

        self.get_t0_value()
        self.get_elements()

    def get_t0_value(self):
        try:
            dct = self.note.doctime
        except AttributeError:
            dct = get_doctime_timex(self.note.note_path)
        t0_value = dct.attrib['value']
        # currently we only care about dates, not hours etc.
        match = re.match('[\d\-]+', t0_value)
        if match:
            self.t0_value = match.group()
        else:
            self.t0_value = 'unknown'

        self.t0_real = self.transform_value(self.t0_value) # real number representation

        self.timex_elements['t0'] = {'type': dct.attrib['type'], 'value': self.t0_value}

    def get_elements(self):
        elements = {}
        for sent in self.note.iob_labels:
            for item in sent: # item example: {'entity_label': 'B_DATE', 'entity_value': '1945-09-02', 'entity_id': 't164', 'entity_type': 'TIMEX3'}
                if item['entity_type'] == 'TIMEX3' and item['entity_label'] == 'B_DATE': # only consider "date", not "duration" now
                    tid = item['entity_id']
                    value = item['entity_value']
                    type = item['entity_label'][2:]
                    elements[tid] = {'value': value, 'type': type}
            self.timex_elements.update(elements)

    def compare_timex_pairs(self):
        timex_ids = sorted(list(self.timex_elements.keys()), key=lambda x: int(x[1:]))
        if not timex_ids:
            print "timex ids not found"

        N = len(timex_ids)
        # for i, item in enumerate(timex_ids):
        #     for j in range(i+1, N):
        #         timex_id_pairs.append((item, timex_ids[j]))
        timex_id_pairs = self.note.timex_pairs

        predictions = []
        failed_pairs = []
        for pair in timex_id_pairs:
            try:
                val1 = self.timex_elements[pair[0]].get('value', '')
                val2 = self.timex_elements[pair[1]].get('value', '')
                label = self.compare_timex_pair(val1, val2)
            except KeyError:  # we can only handle Date type timexes. Others will throw an exception
                label = None
            if label is not None: # only collect classifiable cases
                predictions.append((pair, label))
            else:
                failed_pairs.append(pair)

        return predictions, failed_pairs

    def transform_value(self, val):
        """transform timex value to tuple (start, end)
            start and end are real numbers
        """
        if not val: return None

        if val == 'PRESENT_REF':
            return self.t0_real
        if val == 'PAST_REF': # means recently
            return self.t0_real[0]-1, self.t0_real[0]
        if val == 'FUTURE_REF':
            return self.t0_real[1], self.t0_real[1]+1

        sign = 1
        if val[0:2] == 'BC':
            sign = -1
            val = val[2:]

        numbers = val.split('-')

        try:
            if len(numbers) == 1:
                year = sign * float(numbers[0])
                return year, year+self.YEAR_END

            elif len(numbers) == 2:
                year = sign * float(numbers[0])
                if numbers[1] == 'H1':
                    return year, year+0.5
                elif numbers[1] == 'H2':
                    return year+0.5, year+self.YEAR_END
                elif numbers[1] == 'WXX': # this week
                    return year-3.0/365, year+4.0/365
                elif numbers[1][0] == 'W':
                    week = float(numbers[1][1:]) - 1
                    return year+week*7/365, year+week*7/365
                elif numbers[1] == 'Q1':
                    return year, year+0.25
                elif numbers[1] == 'Q2':
                    return year+0.25, year+0.5
                elif numbers[1] == 'Q3':
                    return year+0.5, year+0.75
                elif numbers[1] == 'Q4':
                    return year+0.75, year+self.YEAR_END
                elif numbers[1] == 'SP':
                    return year+2.0/12, year+5.0/12 # beginning of March to end of May
                elif numbers[1] == 'SU':
                    return year+5.0/12, year+8.0/12 # beginning of June to end of August
                elif numbers[1] == 'FA':
                    return year+8.0/12, year+10.0/12 # beginning of September to end of November
                else:
                    month = float(numbers[1]) - 1
                    return year+month/12, year+month/12+self.MONTH_END

            elif len(numbers) == 3:
                year, s1, s2 = numbers

                if s2 == 'WE': # weekend
                    week = float(s1[1:]) - 1
                    return sign*float(year)+week*7.0/365+5.0/365, sign*float(year)+week*7.0/365+7.0/365

                match = re.match('(\d+)T', s2) # ignore time. only date used.
                if match:
                    day = match.group(1)
                else:
                    day = s2

                month = float(s1) - 1
                day = float(day) - 1
                return sign*float(year) + month/12 + day/365, sign*float(year) + month/12 + day/365

        except ValueError:
            print "unrecognized DATE type:", val
            return None

        print "unrecognized DATE type:", val
        return None

    def compare_timex_pair(self, val1, val2):
        interval1 = self.transform_value(val1)
        interval2 = self.transform_value(val2)
        if not interval1 or not interval2:
            return None
        if interval1[1] < interval2[0]:
            return 'BEFORE'
        if interval1[0] > interval2[1]:
            return 'AFTER'
        if interval1 == interval2:
            return 'SIMULTANEOUS'
        if interval1[0] < interval2[0] and interval1[1] > interval2[1]:
            return 'INCLUDES'
        if interval1[0] > interval2[0] and interval1[1] < interval2[1]:
            return 'IS_INCLUDED'
        return None


def predict_timex_rel(notes):
    labels = []
    pair_index = {}  # {(i, pair) : index}
    index_offset = 0
    for i, note in enumerate(notes):
        time_ref = TimeRefNetwork(note)
        note_predictions, failed_pairs = time_ref.compare_timex_pairs()
        for pair in failed_pairs:
            # if 't0' not in pair:
            #     note.cross_sentence_pairs += failed_pairs  # send them to cross-sentence model
            # else:
            #     note_predictions.append((pair, None))
            note_predictions.append((pair, None))

        n = len(note_predictions)
        note_keys = [(i, pair) for pair, rel in note_predictions]
        note_indexes = [x+index_offset for x in range(n)]
        note_pair_index = dict(zip(note_keys, note_indexes))

        labels += note_predictions
        pair_index.update(note_pair_index)
        index_offset += n

    # only return relations from labels, not pairs
    labels = [x[1] for x in labels]

    return labels, pair_index


