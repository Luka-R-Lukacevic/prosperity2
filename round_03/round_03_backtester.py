import copy
import json
import math
import jsonpickle
import collections
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from datamodel import (Listing, Observation, Order, OrderDepth,
                       ProsperityEncoder, Symbol, Trade, TradingState)

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}
parameters = [30, 375, 10]
parameters = [425, 125, 10]
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
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

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    volume_traded = copy.deepcopy(empty_dict)

    
    trade_at = parameters[0]
    premium = parameters[1]
    volume_divisor = parameters[2]
    
    def calc_next_price_starfruit(self, cache, state: TradingState):
        
        y = np.array(cache)
        x = np.array([state.timestamp - 100 * x for x in range(10, 0, -1)])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        fair_value_regression = m * state.timestamp + c
        
        return int(round(fair_value_regression))


    def values_extract(self, order_dict, buy = 0):
        
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            
            if(buy==0):
                vol *= -1
            
            tot_vol += vol
            
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val


    def calc_orchids_profit_margin(self, import_tariff_obs, transport_fees_obs):
        cost = import_tariff_obs + transport_fees_obs
        
        margin = 1
        if cost >= -1:
            margin = 0.5
        else:
            margin = (-1/2) * cost
        
        return margin


    def compute_orders_amethysts(self, product, state: TradingState):
        
        orders: list[Order] = []
        
        order_depth: OrderDepth = state.order_depths[product]
        
        acc_bid = 10000
        acc_ask = 10000

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT[product]:
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[product]:
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders


    def compute_orders_starfruit(self, product, state: TradingState, new_starfruit_cache):
        
        orders: list[Order] = []
        
        order_depth: OrderDepth = state.order_depths[product]
        
        LIMIT = self.POSITION_LIMIT[product]
        
        if len(new_starfruit_cache) == 10:
            new_starfruit_cache.pop(0)

        vol_ask, bs_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items())))
        vol_bid, bb_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True)), 1)

        starfruit_mid = (bs_starfruit + bb_starfruit) / 2
        new_starfruit_cache.append(starfruit_mid)
        
        n = len(new_starfruit_cache)

        INF = 1e9
        
        starfruit_lb = -INF
        starfruit_ub = INF
        
        if n == 10:
            fair = self.calc_next_price_starfruit(new_starfruit_cache, state)
            starfruit_lb = fair - 1
            starfruit_ub = fair + 1
            
        else:
            if n > 0:
                starfruit_lb = starfruit_mid - 1  #for the first 10 timestamps we use the current mid price as our price prediction
                starfruit_ub = starfruit_mid + 1
        
        acc_bid = starfruit_lb
        acc_ask = starfruit_ub

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return new_starfruit_cache, orders


    def compute_orders_orchids(self, product, state: TradingState):

        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []
        
        position_limit = self.POSITION_LIMIT[product]
        
        conversions = 0
        
        target_inventory = 0
        
        bid_price_obs = state.observations.conversionObservations[product].bidPrice
        ask_price_obs = state.observations.conversionObservations[product].askPrice
        transport_fees_obs = state.observations.conversionObservations[product].transportFees
        export_tariff_obs = state.observations.conversionObservations[product].exportTariff
        import_tariff_obs = state.observations.conversionObservations[product].importTariff	
        sunlight_obs = state.observations.conversionObservations[product].sunlight              
        humidity_obs = state.observations.conversionObservations[product].humidity              
        
        import_price = ask_price_obs + import_tariff_obs + transport_fees_obs
        export_price = bid_price_obs - export_tariff_obs - transport_fees_obs

        conversions += - self.position[product] + target_inventory

        margin = self.calc_orchids_profit_margin(import_tariff_obs, transport_fees_obs)
        
        orders.append(Order(product, int(round(import_price + margin)), - position_limit))
        orders.append(Order(product, int(round(export_price - 2)), position_limit))
        
        return orders, conversions


    def compute_orders_basket(self, order_depth):

        orders = {'CHOCOLATE ' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/self.volume_divisor:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/self.volume_divisor:
                    break

        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - self.premium
        res_buy = mid_price['CHOCOLATE']*4 + mid_price['STRAWBERRIES']*6 + mid_price['ROSES'] + self.premium - mid_price['GIFT_BASKET']



        trade_at = self.trade_at


        if res_buy > trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol)) 

        elif res_sell > trade_at:
            vol = -(self.POSITION_LIMIT['GIFT_BASKET'] + self.position['GIFT_BASKET'])
            logger.print("res_sell: ", res_sell)
            logger.print("vol: ", vol) 
            logger.print("worst_buy['GIFT_BASKET']", worst_buy['GIFT_BASKET'])
            if vol < 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], vol))

        return orders
    

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        # Initialize the method output dict as an empty dict
        orders_result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}
        conversions_result = 0
        
        # Decode traderData dict from previous iteration 
        decoded_dict = {}
        if state.traderData:
            decoded_dict = jsonpickle.decode(state.traderData)
        
        # Write starfruit cache from traderData
        new_starfruit_cache = []
        if "new_starfruit_cache" in decoded_dict.keys():
            new_starfruit_cache = decoded_dict["new_starfruit_cache"]
        
        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        
        '''
        for product in ['AMETHYSTS', 'STARFRUIT', "ORCHIDS"]:
            
            if product == "AMETHYSTS":
                orders = self.compute_orders_amethysts(product, state)
                orders_result[product] += orders
            
            elif product == "STARFRUIT":
                new_starfruit_cache, orders = self.compute_orders_starfruit(product, state, new_starfruit_cache)
                orders_result[product] += orders
            
            elif product == "ORCHIDS":
                orders, conversions = self.compute_orders_orchids(product, state)
                conversions_result += conversions
                orders_result[product] += orders
        '''

        orders = self.compute_orders_basket(state.order_depths)
        if 'GIFT_BASKET' in orders:
            orders_result['GIFT_BASKET'] += orders['GIFT_BASKET']
        if 'CHOCOLATE' in orders:
            orders_result['CHOCOLATE'] += orders['CHOCOLATE']
        if 'STRAWBERRIES' in orders:
            orders_result['STRAWBERRIES'] += orders['STRAWBERRIES']
        if 'ROSES' in orders:
            orders_result['ROSES'] += orders['ROSES']
        
        
        new_dict = {"new_starfruit_cache": new_starfruit_cache}
        trader_data = jsonpickle.encode(new_dict)
        
        
        logger.flush(state, orders_result, conversions_result, trader_data)
        
        return orders_result, conversions_result, trader_data
