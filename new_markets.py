from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import re

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()

Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}

class Trader:
    def __init__(self):
        self.last5k = {key: [] for key in Limits}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        trade_vol = 0
        my_vol = 0
        
        if state.traderData != "":
            numbers = re.findall(r'\d+', state.traderData)
            trade_vol = int(numbers[0])
            my_vol = int(numbers[1])

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}
        for product in state.listings:

            
            Limit = Limits[product]
            if product not in state.market_trades.keys():
                continue
            trades: List[Trade] = state.market_trades[product]
            orders: List[Order] = []

            for trade in trades:
                self.last5k[product].append(trade)
                trade_vol+=trade.quantity

            if product in state.own_trades.keys():
                my_trades: List[Trade] = state.own_trades[product]
                for trade in my_trades:
                    my_vol +=trade.quantity
            
            #average price of last 5k timestamps
            volume = 0
            price = 0
            for trade in self.last5k[product]:
                if trade.timestamp + 15000 < state.timestamp:
                    self.last5k[product].remove(trade)        
                volume += abs(trade.quantity)
                price += abs(trade.quantity) * trade.price
            fair_value_average = price / volume

            #regression
            x = []
            y = []

            for trade in self.last5k[product]:
                if trade.timestamp in x: 
                    continue
                trades_at_timestamp = [tr for tr in self.last5k[product] if tr.timestamp == trade.timestamp]
                x.append(trade.timestamp)
                price_at_timestamp = sum([tr.price for tr in trades_at_timestamp]) / len(trades_at_timestamp)
                y.append(price_at_timestamp)

            x = np.array(x)
            y = np.array(y)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            fair_value_regression = m*state.timestamp + c

            logger.print("Fair price : " + str(fair_value_average))
            logger.print("Fair price regression: " + str(fair_value_regression))

            if product == "AMETHYSTS": fair_value_regression = 10000

            bid = round(fair_value_regression) -2
            ask = bid+4

            if (state.position.get(product, 0) / Limit > 0.5):
                bid -= 1
                ask -= 1
            if (state.position.get(product, 0) / Limit < -0.5):
                bid += 1
                ask += 1

            orders.append(Order(product, bid, Limit - state.position.get(product, 0)))
            orders.append(Order(product, ask, - Limit - state.position.get(product, 0)))
            
            result[product] = orders
    
    
        trader_data = f'The trade volume is {trade_vol} and {my_vol}.'
        

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data