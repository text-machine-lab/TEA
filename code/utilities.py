

def combineLabels(timexLabels, eventLabels, OLabels=[]):
	'''
	combineTimexEventLabels():
		merge event and timex labels into one list, adding instance ids

	@param timexLabels: list of timex labels for entities.
	@param eventLabels: list of event labels for entities. Includes no instances labeled as timexs
	@return: list of dictionaries, with one dictionary for each entity
	'''

	labels = []

# creation time is always t0
	for i, timexLabel in  enumerate(timexLabels):
		label = {"entity_label": timexLabel, "entity_type": "TIMEX3", "entity_id": "t" + str(i+1)}
		labels.append(label)

	for i, eventLabel in enumerate(eventLabels):
		label = {"entity_label": eventLabel, "entity_type": "EVENT", "entity_id": "e" + str(i)}
		labels.append(label)

	for i, Olabel in enumerate(OLabels):
		label = {"entity_label": Olabel, "entity_type": None, "entity_id": None}
		labels.append(label)

	assert len(labels) == len(timexLabels + eventLabels + OLabels)

	return labels
