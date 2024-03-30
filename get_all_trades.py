from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):

        result = {}
        trades = []
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 0
            if product == "AMETHYSTS":
                acceptable_price = 10000
                    
                bid = acceptable_price -1
                ask = bid+2
    
    
                orders.append(Order(product, bid, 20 - state.position.get(product, 0)))
                orders.append(Order(product, ask, -20 - state.position.get(product, 0)))
            
            result[product] = orders

            if product in state.market_trades.keys():
                for trade in state.market_trades[product]:
                    if trade.timestamp + 100 == state.timestamp: #recent trade
                        trades.append((product, trade.price, trade.quantity))
            if product in state.own_trades.keys():
                for trade in state.own_trades[product]:
                    if trade.timestamp + 100 == state.timestamp: #recent trade
                        trades.append((product, trade.price, trade.quantity))

        print(trades)
        traderData = "" 
         
        conversions = None
        return result, conversions, traderData