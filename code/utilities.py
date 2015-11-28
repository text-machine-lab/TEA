

def combineTimexEventLabels(timexLabels, eventLabels):
	'''
	combineTimexEventLabels():
		merge event and timex labels into one list, adding instance ids

	@param timexLabels: list of timex labels for entities. 
	@param eventLabels: list of event labels for entities. Includes no instances labeled as timexs
	@return: list of dictionaries, with one dictionary for each entity
	'''

	labels = []

	eventIndex = 0
	for i in range(0, len(timexLabels) ):
		#add timex label if present
		if timexLabels[i] != 'O':
			label = {"entity_label": timexLabels[i], "entity_type": "TIMEX3", "entity_id": "t" + str(i)}
			labels.append(label)

		#add event label or empty label. Increment event index, as a non-timex entity has been encountered
		else:
			if eventLabels[eventIndex] != 'O':
				label = {"entity_label": eventLabels[eventIndex], "entity_type": "EVENT", "entity_id": "e" + str(i)}
				labels.append(label)
			else:
				label = {"entity_label": 'O', "entity_type": None, "entity_id": None}
				labels.append(label)
			eventIndex += 1
		print i


	print len(labels), len(timexLabels), len(eventLabels)

	assert len(labels) == len(timexLabels)

	return labels
