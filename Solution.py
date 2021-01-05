import numpy as np
def winning_strategy(prices_list):
    print(prices_list)
    cost_price = []
    selling_price= []
    profit_margin = []
    index = prices_list.index((min(prices_list)))
    value = min(prices_list)
    maxprofit = 0
    for i in range(len(prices_list)): #buyingprice
        print("Price list index", prices_list[i])
        for j in range(len(prices_list)): #assumesp
            print(i,j)
            newprofit = prices_list[j] - prices_list[i]
            print(newprofit)
            print("Inital newprofit", newprofit)
            if newprofit > maxprofit:
                print(newprofit)
                print(maxprofit)
                maxprofit = newprofit
                print("maximum profit" , maxprofit)
                if prices_list[j] >= prices_list(i):
                    newprofit = prices_list[j] - prices_list[i]
                print(newprofit)

            #comb = prices_list[i] * prices_list[j]


    for i in range(index+1,len(prices_list)):
        result = max (result, prices_list[i])
        cost_price = (selling_price * 100 ) / 100 + profit_margin
        selling_price = cost_price + profit_margin



    return True
    #return best_buying_day, best_selling_day

winning_strategy(prices_list=[20,30,40,50,10])
