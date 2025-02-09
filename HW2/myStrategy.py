def myStrategy(pastPriceVec, currentPrice):

	import numpy as np

	# Set best parameters
	windowSize = 25
	alpha = 17
	beta = 83
	action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action

	dataLen = len(pastPriceVec)  # Length of the data vector

	# If not enough data, return default action
	if dataLen < windowSize: 
		ma = np.mean(pastPriceVec)
		return action

	rsi_data = pastPriceVec[-windowSize:]
	nege_data = []
	posi_data = []

	for i in range(windowSize - 1):  # Loop to windowSize - 1 to prevent index out of bounds
		diff = rsi_data[i+1] - rsi_data[i]
		if diff >= 0:
			posi_data.append(diff)
		else:
			nege_data.append(abs(diff))  # Use absolute value for negative differences

	# Ensure there's no division by zero
	if len(posi_data) == 0:
		posi_avg = 0
	else:
		posi_avg = sum(posi_data) / windowSize

	if len(nege_data) == 0:
		nege_avg = 0
	else:
		nege_avg = sum(nege_data) / windowSize

	# Compute RSI
	if posi_avg + nege_avg == 0:  # Avoid division by zero
		return action
	else:
		RSI = posi_avg / (posi_avg + nege_avg)

	# Determine action
	RSI_percentage = RSI * 100

	if  RSI_percentage <= alpha:
		action = 1
	elif RSI_percentage > beta:
		action = -1

	return action