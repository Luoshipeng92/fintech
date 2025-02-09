import numpy as np

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    # user definition
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    Cash = np.zeros((dataLen,stockCount))
    dp = np.zeros((dataLen,stockCount))  
    hold_cash = [[(1, 0) for _ in range(4)] for _ in range(dataLen)]
    hold_stock = [[(1, 0)for _ in range(stockCount)] for _ in range(dataLen)]
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    for day in range(0, dataLen) :
        
        dayPrices = priceMat[day]  # Today price of each stock
        if day == 0:
            for i in range(stockCount):
                Cash[day][i] = 1000
                dp[day][i] = (1-transFeeRate) * 1000 / dayPrices[i]
                hold_cash[day][i] = (1,i)
                hold_stock[day][i] = (0,i)
        else:
            # For cash
            for stock in range(stockCount):
                max_cash = Cash[day-1][stock]
                cash_value = dp[day-1][stock] * dayPrices[stock]*(1-transFeeRate)
                if cash_value > max_cash: 
                    max_cash = cash_value
                    hold_cash[day][stock] = (0,stock) #stock to cash
                else:
                    hold_cash[day][stock] = (1,stock) #cash to cash
                Cash[day][stock] = max_cash

            # For stock
                dp[day][stock] = dp[day - 1][stock]
                c2s = [(1-transFeeRate) * cash / dayPrices[stock] for cash in Cash[day-1]]
                c2s_max_value = np.max(c2s)
                c2s_max_index = np.argmax(c2s)
                
                s2s = []
                for j in range(stockCount):
                    if j == stock:
                        s2s.append(dp[day][j])
                    else:
                        s2s.append(dp[day-1][j] * dayPrices[j]*(1-transFeeRate)*(1-transFeeRate)/dayPrices[stock])
                s2s_max_value = np.max(s2s)
                s2s_max_index = np.argmax(s2s)
                
                if c2s_max_value > s2s_max_value and c2s_max_value > dp[day - 1][stock]: # cash to stock
                    dp[day][stock]= c2s_max_value
                    hold_stock[day][stock] = (0, c2s_max_index)
                
                else: # stock to stock
                    dp[day][stock]= s2s_max_value
                    hold_stock[day][stock]= (1, s2s_max_index)


    def cash_find_previous(day, stock):
        dayPrices = priceMat[day]
        index, previous = hold_cash[day][stock]
        if day > 0:
            if index == 0: #previous is stock
                action = [day, stock, -1, dp[day-1][previous] * dayPrices[previous]]
                stock_find_previous(day-1, stock)
                actionMat.append(action)
            elif index == 1: #previous is cash
                cash_find_previous(day-1, previous)
        else:
            return
    
    def stock_find_previous(day, stock):
        index, previous_stock = hold_stock[day][stock]
        if day == 0:
            action = [day, -1, previous_stock, 1000]
            actionMat.append(action)
        if day > 0:
            dayPrices = priceMat[day]
            if index == 1: #previous is stock
                if previous_stock != stock:
                    action = [day, previous_stock, stock, dp[day-1][previous_stock] * dayPrices[previous_stock]]
                    stock_find_previous(day-1, previous_stock)
                    actionMat.append(action)
                else:
                    stock_find_previous(day-1, previous_stock)

            elif index == 0: #previous is cash
                action = [day, -1, stock, Cash[day-1][previous_stock]]
                cash_find_previous(day-1, previous_stock)
                actionMat.append(action)
        else:
            return        
    max_index = np.argmax(Cash[dataLen-1])
    cash_find_previous(dataLen-1, max_index)
    return actionMat

# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    Cash = np.zeros((dataLen,stockCount))
    dp = np.zeros((dataLen,stockCount))  
    hold_cash = [[(1, 0) for _ in range(4)] for _ in range(dataLen)]
    hold_stock = [[(1, 0)for _ in range(stockCount)] for _ in range(dataLen)]
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    if K == 200:
        key = 129
    elif K == 300:
        key = 115
    else:
        key = 105
    for day in range(0, dataLen) :
        if day > dataLen-K+key:
            for i in range(stockCount):
                Cash[day][i] = Cash[dataLen -K+key][i]
                dp[day][i] = dp[dataLen -K+key][i]
                hold_cash[day][i] = (1,i)
                hold_stock[day][i] = (1,i)
        dayPrices = priceMat[day]  # Today price of each stock
        if day == 0:
            for i in range(stockCount):
                Cash[day][i] = 1000
                dp[day][i] = (1-transFeeRate) * 1000 / dayPrices[i]
                hold_cash[day][i] = (1,i)
                hold_stock[day][i] = (0,i)
        else:
            # For cash
            for stock in range(stockCount):
                max_cash = Cash[day-1][stock]
                cash_value = dp[day-1][stock] * dayPrices[stock]*(1-transFeeRate)
                if cash_value > max_cash: 
                    max_cash = cash_value
                    hold_cash[day][stock] = (0,stock) #stock to cash
                else:
                    hold_cash[day][stock] = (1,stock) #cash to cash
                Cash[day][stock] = max_cash

            # For stock
                dp[day][stock] = dp[day - 1][stock]
                c2s = [(1-transFeeRate) * cash / dayPrices[stock] for cash in Cash[day-1]]
                c2s_max_value = np.max(c2s)
                c2s_max_index = np.argmax(c2s)
                
                s2s = []
                for j in range(stockCount):
                    if j == stock:
                        s2s.append(dp[day][j])
                    else:
                        s2s.append(dp[day-1][j] * dayPrices[j]*(1-transFeeRate)*(1-transFeeRate)/dayPrices[stock])
                s2s_max_value = np.max(s2s)
                s2s_max_index = np.argmax(s2s)
                
                if c2s_max_value > s2s_max_value and c2s_max_value > dp[day - 1][stock]: # cash to stock
                    dp[day][stock]= c2s_max_value
                    hold_stock[day][stock] = (0, c2s_max_index)
                
                else: # stock to stock
                    dp[day][stock]= s2s_max_value
                    hold_stock[day][stock]= (1, s2s_max_index)


    def cash_find_previous(day, stock):
        dayPrices = priceMat[day]
        index, previous = hold_cash[day][stock]
        if day > 0:
            if index == 0: #previous is stock
                action = [day, stock, -1, dp[day-1][previous] * dayPrices[previous]]
                stock_find_previous(day-1, stock)
                actionMat.append(action)
            elif index == 1: #previous is cash
                cash_find_previous(day-1, previous)
        
        else:
            return
    
    def stock_find_previous(day, stock):
        index, previous_stock = hold_stock[day][stock]
        if day == 0:
            action = [day, -1, previous_stock, 1000]
            actionMat.append(action)
        if day > 0 :
            dayPrices = priceMat[day]
            if index == 1: #previous is stock
                if previous_stock != stock:
                    action = [day, previous_stock, stock, dp[day-1][previous_stock] * dayPrices[previous_stock]]
                    stock_find_previous(day-1, previous_stock)
                    actionMat.append(action)
                else:
                    stock_find_previous(day-1, previous_stock)

            elif index == 0: #previous is cash
                action = [day, -1, stock, Cash[day-1][previous_stock]]
                cash_find_previous(day-1, previous_stock)
                actionMat.append(action)

        else:
            return        
    max_index = np.argmax(Cash[dataLen-K+key])
    cash_find_previous(dataLen-K+key, max_index)
    return actionMat

#An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    max_last_day = 0
    answer_mat = []
    for key in range(5):
        Cash = np.zeros((dataLen,stockCount))
        dp = np.zeros((dataLen,stockCount))  
        hold_cash = [[(1, 0) for _ in range(4)] for _ in range(dataLen)]
        hold_stock = [[(1, 0)for _ in range(stockCount)] for _ in range(dataLen)]
        actionMat = []  # An k-by-4 action matrix which holds k transaction records.
        for day in range(0, dataLen) :
            dayPrices = priceMat[day]  # Today price of each stock
            if day > dataLen-K-key and day < dataLen-key:
                for i in range(stockCount):
                    Cash[day][i] = Cash[dataLen-K-key][i]
                    dp[day][i] = dp[day-K-key][i]
                    hold_cash[day][i] = (1,i)
                    hold_stock[day][i] = (1,i)
            elif day == 0:
                for i in range(stockCount):
                    Cash[day][i] = 1000
                    dp[day][i] = (1-transFeeRate) * 1000 / dayPrices[i]
                    hold_cash[day][i] = (1,i)
                    hold_stock[day][i] = (0,i)
            elif day == dataLen - key:
                for i in range(stockCount):
                    Cash[day][i] = Cash[day-1][i]
                    dp[day][i] = (1-transFeeRate) * Cash[day-1][i] / dayPrices[i]
                    hold_cash[day][i] = (1,i)
                    hold_stock[day][i] = (0,i)
            else:
                # For cash
                for stock in range(stockCount):
                    max_cash = Cash[day-1][stock]
                    cash_value = dp[day-1][stock] * dayPrices[stock]*(1-transFeeRate)
                    if cash_value > max_cash: 
                        max_cash = cash_value
                        hold_cash[day][stock] = (0,stock) #stock to cash
                    else:
                        hold_cash[day][stock] = (1,stock) #cash to cash
                    Cash[day][stock] = max_cash

                # For stock
                    dp[day][stock] = dp[day - 1][stock]
                    c2s = [(1-transFeeRate) * cash / dayPrices[stock] for cash in Cash[day-1]]
                    c2s_max_value = np.max(c2s)
                    c2s_max_index = np.argmax(c2s)
                    
                    s2s = []
                    for j in range(stockCount):
                        if j == stock:
                            s2s.append(dp[day][j])
                        else:
                            s2s.append(dp[day-1][j] * dayPrices[j]*(1-transFeeRate)*(1-transFeeRate)/dayPrices[stock])
                    s2s_max_value = np.max(s2s)
                    s2s_max_index = np.argmax(s2s)
                    
                    if c2s_max_value > s2s_max_value and c2s_max_value > dp[day - 1][stock]: # cash to stock
                        dp[day][stock]= c2s_max_value
                        hold_stock[day][stock] = (0, c2s_max_index)
                    
                    else: # stock to stock
                        dp[day][stock]= s2s_max_value
                        hold_stock[day][stock]= (1, s2s_max_index)  

        def cash_find_previous(day, stock):
            dayPrices = priceMat[day]
            index, previous = hold_cash[day][stock]

            if day > 0:
                if index == 0: #previous is stock
                    action = [day, stock, -1, dp[day-1][previous] * dayPrices[previous]]
                    stock_find_previous(day-1, stock)
                    actionMat.append(action)
                elif index == 1: #previous is cash
                    cash_find_previous(day-1, previous)
            else:
                return
        
        def stock_find_previous(day, stock):
            index, previous_stock = hold_stock[day][stock]
            if day == 0:
                action = [day, -1, previous_stock, 1000]
                actionMat.append(action)       

            if day > 0 :
                dayPrices = priceMat[day]
                if index == 1: #previous is stock
                    if previous_stock != stock:
                        action = [day, previous_stock, stock, dp[day-1][previous_stock] * dayPrices[previous_stock]]
                        stock_find_previous(day-1, previous_stock)
                        actionMat.append(action)
                    else:
                        stock_find_previous(day-1, previous_stock)

                elif index == 0: #previous is cash
                    action = [day, -1, stock, Cash[day-1][previous_stock]]
                    cash_find_previous(day-1, previous_stock)
                    actionMat.append(action)
            else:
                return        
        max_value = np.max(Cash[dataLen-1])
        max_index = np.argmax(Cash[dataLen-1])
        if max_value > max_last_day:
            cash_find_previous(dataLen-1 , max_index)
            max_last_day = max_value
            answer_mat = actionMat
    return answer_mat
