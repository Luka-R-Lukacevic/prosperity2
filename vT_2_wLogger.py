from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol, Trade, Listing, Observation
import math as m
import statistics as stat
import json
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


class Trader:

    def __init__(self):
        self.product_parameters = { 'AMETHYSTS':{
                                    'inventory_limit': 20, 'fair_price' : 10000,
                                    'lower_inventory_limit': -20, 'upper_inventory_limit': 20,
                                    },
                                
                                    'STARFRUIT':{
                                    'inventory_limit': 20, 'fair_price' : 5000,
                                    'lower_inventory_limit': -20, 'upper_inventory_limit': 20,
                                    }
                                }
    
    def initialize(self, state: TradingState, product):
        """
        Initialize and calculate all needed variables on every call for every tradable object
        """
        inventory_limit = self.product_parameters[product]['inventory_limit']
        order_depth: OrderDepth = state.order_depths[product]   # Retrieve the Order Depth containing all the market BUY and SELL orders
        lob_buy_volume_total = m.fabs(m.fsum(order_depth.buy_orders.values()))       # Returns the absolute value of the aggregated buy order volumes of the orderbook (lob) --> positive
        lob_sell_volume_total = m.fabs(m.fsum(order_depth.sell_orders.values()))     # Returns the absolute value of the aggregated sell order volumes of the orderbook (lob) --> positive
        lob_buy_strikes = list(order_depth.buy_orders.keys())        # Returns a list of all unique strikes where at least 1 buy order has been submitted to the orderbook (lob) --> positive & low to high strikes
        lob_sell_strikes = list(order_depth.sell_orders.keys())      # Returns a list of all unique strikes where at least 1 sell order has been submitted to the orderbook (lob) --> positive & low to high strikes
        lob_buy_volume_per_strike = list(order_depth.buy_orders.values())        # Returns a list of volumes for the lob_buy_strikes --> positive & low to high strikes
        lob_sell_volume_per_strike = [-x for x in list(order_depth.sell_orders.values())]        # Returns a list of volumes for the lob_sell_strikes --> positive & low to high strikes
        lob_buy_strikes_w_vol = [lob_buy_strikes[lob_buy_volume_per_strike.index(x)] for x in lob_buy_volume_per_strike for i in range(x)]      # Returns a list of all buy strikes with the amount of strikes in the list given by their quantity in lob_buy_volume_per_strike
        lob_sell_strikes_w_vol = [lob_sell_strikes[lob_sell_volume_per_strike.index(x)] for x in lob_sell_volume_per_strike for i in range(x)]      # Returns a list of all sell strikes with the amount of strikes in the list given by their quantity in lob_sell_volume_per_strike
        lob_average_buy = stat.fmean(lob_buy_strikes_w_vol)
        lob_average_sell = stat.fmean(lob_sell_strikes_w_vol)
        lob_buy_quantity = sum(lob_buy_volume_per_strike)
        lob_sell_quantity = sum(lob_sell_volume_per_strike) 
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2 
        
        #lob_all_strikes_w_vol = lob_buy_strikes_w_vol + lob_sell_strikes_w_vol
        #current_timestamp = state.timestamp

        return (lob_average_buy, lob_average_sell, lob_buy_quantity, lob_sell_quantity, lob_buy_strikes, 
                lob_sell_strikes, lob_buy_volume_per_strike, lob_sell_volume_per_strike, inventory_limit,
                lob_buy_volume_total, lob_sell_volume_total,best_bid,mid_price,best_ask)

        
    def calculate_bid_ask(self, product: str, smart_price: float, lob_buy_strikes: list, lob_sell_strikes: list):
        """
        Calculates the the best available Bid/Ask price to maximize profit.
        Works by undercutting the competition and placing bid/ask around the smart price.
        """
        
        for idx, bid_strike in enumerate(lob_buy_strikes):
            if smart_price > bid_strike and abs(smart_price - bid_strike) > 1:
                smart_price_bid = bid_strike + 1
                break
            elif idx+1 == len(lob_buy_strikes):
                smart_price_bid = m.floor(smart_price) - 1
                break
        
        for idx, ask_strike in enumerate(lob_sell_strikes):
            if smart_price < ask_strike and abs(smart_price - ask_strike) > 1:
                smart_price_ask = ask_strike - 1
                break
            elif idx+1 == len(lob_sell_strikes):
                smart_price_ask = m.ceil(smart_price) + 1
                break
        
        while smart_price_ask in lob_sell_strikes and smart_price_ask - 1 >= smart_price:
            smart_price_ask -= 1
        
        while smart_price_bid in lob_buy_strikes and smart_price_bid + 1 <= smart_price:
            smart_price_bid += 1
        
        
        """
        Don't buy Amethysts over fair price of 10,000 and don't sell under either.
        Definitely benificial to use but less flexible for possible price movements.
        """
        if product == "AMETHYSTS":
            if smart_price_bid > self.product_parameters[product]['fair_price']:
                smart_price_bid = self.product_parameters[product]['fair_price']
            elif smart_price_ask < self.product_parameters[product]['fair_price']:
                smart_price_ask = self.product_parameters[product]['fair_price']
        
        return smart_price_bid, smart_price_ask
    
    
    def calculate_smart_price(self, lob_average_buy: float, lob_average_sell: float, lob_buy_quantity: int, lob_sell_quantity: int) -> float:
        """
        Calculates the "Smart Price" described in "Machine Learning for Market Microstructure and High Frequency Trading" - Michael Kearns
        """
        smart_price = ((lob_average_buy *  lob_sell_quantity) + (lob_average_sell * lob_buy_quantity) ) / (lob_buy_quantity + lob_sell_quantity)
        
        return smart_price
    
    
    def calculate_available_buy_and_sell(self, product, inventory_limit, initial_inventory, mod_1_buy_volume, mod_1_sell_volume):
        """
        Calculates the buy and sell orders still available taking the module 1 orders into account
        """
        upper_bound = self.product_parameters[product]['upper_inventory_limit']
        lower_bound = self.product_parameters[product]['lower_inventory_limit']

        if min(upper_bound,inventory_limit) - initial_inventory - mod_1_buy_volume < 0:
            avail_buy_orders = 0
        else:
            avail_buy_orders = min(upper_bound,inventory_limit) - initial_inventory - mod_1_buy_volume
        if abs(max(lower_bound, -inventory_limit)) + initial_inventory - mod_1_sell_volume < 0:
            avail_sell_orders = 0
        else:
            avail_sell_orders = abs(max(lower_bound, -inventory_limit)) + initial_inventory - mod_1_sell_volume
        
        return round(avail_buy_orders), round(avail_sell_orders)
    
    
    def get_current_inventory(self, state, product, position_changes):
        """
        Just to clean up the trade_logic and make it more readable. This gets the current inventory
        """
        
        initial_inventory = state.position.get(product,0)  
        current_inventory = initial_inventory + position_changes

        return current_inventory
    
    
    def module_1_order_tapper(self, lob_buy_strikes, lob_sell_strikes, lob_buy_volume_per_strike, lob_sell_volume_per_strike, initial_inventory, inventory_limit, product, new_bid, new_ask, smart_price):
        """
        Takes the newly calculated bids and asks and checks if there are buy orders above the calculated bid or sell orders below the newly calculated ask. 
        This could also be done for the Smart Price and not for the bids and asks but this could result in resulting offers in between the bid and ask which takes away volume from the market making algo.
        Will need to backtest if either this method or using the Smart Price instead gives better results.
        """
        buy_volume = 0 
        sell_volume = 0
        buy_volume_total = 0
        sell_volume_total = 0
        new_orders: list[Order] = []
        for strike in lob_sell_strikes:
            if strike < new_ask:
                buy_volume = lob_sell_volume_per_strike[lob_sell_strikes.index(strike)]
                if abs(initial_inventory + buy_volume_total + buy_volume) <= inventory_limit:
                    new_orders.append(Order(product, strike, buy_volume))
                    buy_volume_total += buy_volume
                else:
                    buy_volume = abs(inventory_limit - initial_inventory - buy_volume_total)
                    new_orders.append(Order(product, strike, buy_volume))
                    buy_volume_total += buy_volume
                    break
        
        for strike in lob_buy_strikes:
            if strike > new_bid:
                sell_volume = lob_buy_volume_per_strike[lob_buy_strikes.index(strike)]
                if abs(initial_inventory - sell_volume -sell_volume_total) <= inventory_limit:
                    new_orders.append(Order(product, strike, -sell_volume))
                    sell_volume_total += sell_volume
                else:
                    sell_volume = abs(initial_inventory + inventory_limit - sell_volume_total)
                    new_orders.append(Order(product, strike, -sell_volume))
                    sell_volume_total += sell_volume
                    break

        return (new_orders, buy_volume_total, sell_volume_total)

    def module_2_market_maker(self, smart_price: float, product: str, smart_price_bid: int, smart_price_ask: int, avail_buy_orders: int, avail_sell_orders: int):
        """
        Module for Market Making. This takes the bid/ask of the smart price and places orders at them.
        """
        new_orders: list[Order] = []
        
        new_orders.append(Order(product, (smart_price_bid), avail_buy_orders))
        new_orders.append(Order(product, (smart_price_ask), -avail_sell_orders))
        
        return (new_orders)
    
    
    def trade_logic(self, product:str, state: TradingState, market_variables: list):
        
        #UPDATE THE CURRENT INVENTORY
        current_inventory = self.get_current_inventory(state, product, 0)
        
        #DEFINE AND GET ALL NEEDED ORDERBOOK DATA
        (lob_average_buy, 
        lob_average_sell, 
        lob_buy_quantity, 
        lob_sell_quantity, 
        lob_buy_strikes, 
        lob_sell_strikes, 
        lob_buy_volume_per_strike, 
        lob_sell_volume_per_strike, 
        inventory_limit,
        lob_buy_volume_total,
        lob_sell_volume_total,
        best_bid,
        mid_price,
        best_ask) = market_variables
        
        
        #CHECKS IF THERE ARE ORDERS ON BOTH SIDES OF THE ORDERBOOK
        if lob_buy_volume_total > 0 and lob_sell_volume_total > 0:
            
            #CALC SMART PRICE
            smart_price = self.calculate_smart_price(lob_average_buy, 
                                                    lob_average_sell, 
                                                    lob_buy_quantity, 
                                                    lob_sell_quantity)
            
            #CALC BID/ASK
            smart_price_bid, smart_price_ask = self.calculate_bid_ask(product, smart_price, lob_buy_strikes, lob_sell_strikes)
            
            
            #MOD1 BUY AND SELL ORDERS 
            mod_1_new_orders, mod_1_buy_volume, mod_1_sell_volume = self.module_1_order_tapper(lob_buy_strikes, 
                                                                                                lob_sell_strikes, 
                                                                                                lob_buy_volume_per_strike, 
                                                                                                lob_sell_volume_per_strike, 
                                                                                                current_inventory, 
                                                                                                inventory_limit, 
                                                                                                product, 
                                                                                                smart_price_bid, 
                                                                                                smart_price_ask,
                                                                                                smart_price
                                                                                                )

            
            #AVAILABLE BUYS AND SELLS AFTER MOD 1 ORDERS ARE EXECUTED
            avail_buy_orders, avail_sell_orders = self.calculate_available_buy_and_sell(product, inventory_limit, current_inventory, mod_1_buy_volume, mod_1_sell_volume)
            
            #CALCULATE CURRENT INVENTORY WITH MOD 1 ORDERS ALREADY INCLUDED
            current_inventory = self.get_current_inventory(state, product, (mod_1_buy_volume + mod_1_sell_volume))
            
            #MOD2 BUY AND SELL ORDERS
            mod_2_new_orders = self.module_2_market_maker(smart_price,
                                                        product, 
                                                        smart_price_bid, 
                                                        smart_price_ask, 
                                                        avail_buy_orders, 
                                                        avail_sell_orders
                                                        )
            
            #RETURN ALL ORDER DATA AND THE DATA NEEDED FOR VISUALIZATION
            return mod_1_new_orders, mod_2_new_orders, best_bid,mid_price, best_ask, smart_price_bid, smart_price, smart_price_ask
        
        
        elif lob_buy_volume_total > 0 and not lob_sell_volume_total > 0:
            """
            Only orders on the buy side of the book.
            """
            
            ridiculous_sell_order = []
            avail_buy_orders, avail_sell_orders = self.calculate_available_buy_and_sell(product, inventory_limit, current_inventory, 0, 0)
            
            sell_price = round(best_bid * 1.005)
            
            ridiculous_sell_order.append(Order(product, sell_price, -avail_sell_orders))
            
            return ridiculous_sell_order
        
        elif not lob_buy_volume_total > 0 and lob_sell_volume_total > 0:
            """
            Only orders on the sell side of the book.
            """
            
            ridiculous_buy_order = []
            avail_buy_orders, avail_sell_orders = self.calculate_available_buy_and_sell(product, inventory_limit, current_inventory, 0, 0)
            
            buy_price = round(best_ask * 0.995)
            
            ridiculous_buy_order.append(Order(product, buy_price, avail_buy_orders))
            
            return ridiculous_buy_order

        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        #INITIALIZE THE OUTPUT DICT AS AN EMPTY DICT
        total_transmittable_orders = {}


        #ITERATE OVER ALL AVAILABLE PRODUCTS IN THE ORDER DEPTHS
        for product in state.order_depths.keys():
            if product in ['STARFRUIT','AMETHYSTS']:
                
                #DEFINE AND GET ALL ORDER BOOK DATA
                market_variables = self.initialize(state, product)
                
                #EXECUTE THE TRADE LOGIC AND OUTPUTS ALL ORDERS + DATA NEEDED FOR VISUALIZATION
                mod1, mod2, best_bid, mid_price, best_ask, smart_price_bid, smart_price, smart_price_ask = self.trade_logic(product, state, market_variables)
                
                #ADDS ALL ORDERS TO BE TRANSMITTED TO THE EMPTY DICT
                total_transmittable_orders[product] = mod1 + mod2
                
        
        
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        # Any Python variable could be serialised into string with jsonpickle library and deserialised on the next call based on TradingState.traderData property. Container will not interfere with the content. 
        traderData = "" 
        
		# Sample conversion request. Will need this in later rounds 
        conversions = 0
        
        #RETURNS ALL ORDER DATA TO THE ENGINE
        
        
        logger.flush(state, total_transmittable_orders, conversions, traderData)
        return total_transmittable_orders, conversions, traderData