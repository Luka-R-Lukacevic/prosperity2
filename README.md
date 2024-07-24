# Aruba Capital
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


In this repo we present ideas and code for the second IMC Prosperity competition, hosted in 2024. Our team, Aruba Capital, finished 22nd globally out of more then 2800 active competitor teams, placing us in the top 1%.

![Confirmation](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Confirmation%20IMC.jpeg)

## Team members

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jaJann312">
          <img src="https://avatars.githubusercontent.com/u/49684458?v=4" width="100px;" alt="Jannik Proff"/>
          <br /><sub><b>Jannik Proff</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/jproff/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/JGGrosse">
          <img src="https://avatars.githubusercontent.com/u/142249387?v=4" width="100px;" alt="Janek Große"/>
          <br /><sub><b>Janek Große</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/janek-grosse/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/Luka-R-Lukacevic">
          <img src="https://avatars.githubusercontent.com/u/125273166?v=4" width="100px;" alt="Luka Lukačević"/>
          <br /><sub><b>Luka Lukačević</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/luka-lukačević/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/PaulHenrik">
          <img src="https://avatars.githubusercontent.com/u/19336571?v=4" width="100px;" alt="Paul Heilmann"/>
          <br /><sub><b>Paul Heilmann</b></sub></a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/tinotil">
          <img src="https://avatars.githubusercontent.com/u/35002593?v=4" width="100px;" alt="Constantin Schott"/>
          <br /><sub><b>Constantin Schott</b></sub></a>
      </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 🐚 What is prosperity?

"Prosperity is a 15-day long trading challenge happening somewhere in a near - utopian - future. You’re in control of an island in an archipelago and your goal is to bring your island to prosperity. You do so by earning as many SeaShells as possible; the main currency in the archipelago. The more SeaShells you earn, the more your island will prosper. 

During your 15 days on the island, your trading abilities will be tested through a variety of trading challenges. It’s up to you to develop a successful trading strategy. You will be working on a Python script to handle algorithmic trades on your behalf. Every round you will also be confronted with a manual trading challenge. Your success depends on both these algorithmic and manual trades." (IMC description)

There were over 2800 active teams, tasked with algorithmically trading various products, such as amethysts, starfruit, orchids, coconuts, and more, with the goal of maximizing seashells: the underlying currency of our island.

In round 1, we began trading amethysts and starfruit. With each subsequent round, additional products were introduced. Our trading algorithm was assessed at the end of each round by comparing its performance to that of bot participants in the marketplace. We could attempt to predict the bots' behavior using historical data. Our PNL from this evaluation was then compared to that of other teams.

Aside from the main focus on algorithmic trading, the competition included manual trading challenges in each round. These challenges varied significantly, but manual trading ultimately contributed only a small fraction to our overall PNL.

For more details on the algorithmic trading environment and additional context about the competition, please refer to the [Prosperity 2 Wiki](https://imc-prosperity.notion.site/Prosperity-2-Wiki-fe650c0292ae4cdb94714a3f5aa74c85).

## Organization

This repository contains all of our code–including internal tools, research notebooks, raw data and backtesting logs, and all versions of our algorithmic trader. The repository is organized by round.

<details>
<summary><h2>Tools </h2></summary>

Instead of building our tools in-house, we decided to leverage the open-source wizardry of  [Jasper van Merle](https://github.com/jmerle). His tools provided the foundation we needed, allowing us to tailor our focus on other areas of development. We utilized his two main tools: a backtester and a visualiser.

### Backtester

We realized we needed a comprehensive backtesting environment very early on. After going after that ourselves with not a lot of success, fortunately, Jasper van Merle's [backtester](https://github.com/jmerle/imc-prosperity-2-backtester) was released to take in historical data and a trading algorithm. With the historical data, it would construct all the necessary information (replicating the actual trading environment perfectly) that our trading algorithm needed, input it into our trading algorithm, and receive the orders that our algorithm would send. Then, it would match those orders to the orderbook to generate trades. After running, the backtester would create a log file in the exact same format as the Prosperity website, that the visualiser was then able to visualise.


![Backtested PNL](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Backtester%20Image.jpeg)


### Visualiser

Jasper van Merle's [visualizer](https://jmerle.github.io/imc-prosperity-2-visualizer/?/visualizer) visualiser was an immense tool for us that provided a powerful and flexible way to analyze our trading data, helping us to identify and rectify issues, and ultimately improve our trading strategies. 


![Visualiser in Action](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Visualiser%20Image.png)


</details>
<details>
<summary><h2>Round 1️⃣</h2></summary>

In round 1, we had access to two symbols to trade: amethysts and starfruit. 

### Amethysts
Amethysts were fairly simple, as the fair price clearly never deviated from 10,000. As such, we wrote our algorithm to trade against bids above 10,000 and asks below 10,000. Besides taking orders, our algorithm also would market-make, placing bids and asks below and above 10,000, respectively.

### Starfruit

Starfruits were an asset with an orderbook limit of 20 (as were amethysts). Here the price fluctuated much more though, usually up to hundreds of seashells. Also, notice that the spread is pretty wide (around 6-7 consistently, which is much more then for the other products).

![Starfruit](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Starfruit.jpeg)

This opened up the opportunity for market making, provided one had a good price estimate. After trying lots of things we concluded that there was no additional information in knowing the whole price history (in comparison to just the current orderbook). In mathematical terms you could say the prices followed a discrete-time Markov process. As a small digression, in the Black-Scholes (BS) model the assumed SDE that leads to the formula also necessitates the Markov property, we will see more of the BS formula later in round 4.

Still, while we concluded there was basically no point in taking complicated history into account for the fair price, we still had the problem that we could not just use the mid-price (average of highest bid and lowest ask) as our price estimate, since if there are good trades in the orderbook for us, then these will necessarily be either exceptionally high bids or exceptionally low asks. One could then use past history to get a better fair estimate. This works fine, but something else worked even better. Essentially one could see that in the orderbook there were usually bids and ask that had high volume and were around 6-7 apart and then some small deviant orders (in real markets these would be called micro-noise).

![Starfruit micro noise](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Starfruit%20orderbook.jpeg)

A nifty solution came from last years second place, the [Stanford Cardinals](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal/tree/main). The mid-price estimate is simply the mid-price by lowest-bid and highest-ask. In this market this eliminates the micro-noise, allowing us to pick off the bad orders.

After the first round we were in the 70s but with relatively small distance to the lead.


</details>

<details>
<summary><h2>Round 2️⃣</h2></summary>
  
### Orchids
In round 2, orchids were introduced. While they could be traded on our home market just like amethysts and starfruits, there was another market (the "south archipelago") where we were able to trade orchids. Trades executed on that foreign market however were subject to import/export tariffs as well as shipping costs, all of which changed over time. We were only allowed to do trades with the south archipelago however that would bring our position closer to zero (i.e. whenever we were long we could export, whenever we were short we could import). They also gave us historical graphs for sunlight and humidity levels which we were told would influence the orchids' growth/availability. They even gave us exact ranges of values of sunlight/humidity for which the growth would be perfect and how it deviates outside of these ranges. This was however very unclear as they for example failed to provide a unit for the sunlight and gave contradictory information on it in the discord channel.

We started by investigating whether there was any relation between the orchids' price and sunlight/humidity and found a loose correlation. Position-taking on that however gave us only minimal profits. This fact combined with the very vague definition led us to believe that the key in this round was not to be found in the weather (which turned out to be probably true as we have not heard of any successful team using sunlight/humidity at all).

Next up, we tried simple market making on our home island which failed miserably. Experimenting a bit made us realize however that - even though the order book was notoriously empty this round - there was a big buyer that would regularly take all our sell orders. Realizing that we were actually paid the tariffs when importing from the south archipelago, we saw that cross-exchange market making was the way to go: We would sell on our home market to the big buyer and immediately import the same quantity (often 100 which was the position limit) to cancel our position. That way, we were able to trade an incredible volume and our profits on the website skyrocketed to 80-100k. Now that we had the basic strategy (which ironically was implemented in two lines of code), we just had to do some fine-tuning of the only parameter in our code: What was the perfect price for our sell orders on the home market that would maximize volume*profit per share. We realized that this depended heavily on the import tariffs and the shipping costs and also that our strategy was only so profitable because of the incredibly high import tariffs (which we were paid) during the last day. In the end we came up with an algorithm that dynamically adapts the sell price to the current tariffs/shipping costs.

After this round we were ranked 48th. We probably could have gotten a higher rank had we done the optimization of the sell price at times of semi-low import tariffs more carefully. For low import tariffs we realized that there was no profit to be made from this strategy which led us to implement sunlight/humidity based position taking in that case in subsequent rounds.

</details>
<details>
<summary><h2>Round 3️⃣</h2></summary>
  
Gift baskets, chocolate, roses, and strawberries were introduced in round 3, where a gift basket consisted of 4 chocolate bars, 6 strawberries, and a single rose. This round, we mainly traded the gift basket on the signal of the chocolate, the strawberries and the rose minus a premium. We assumed the price of the basket actually trails the prices of the other assets (similar to the Cardinal's strategy last year), the only change we made in our strategy was that we compared not the difference of these two assets but instead the ratio (with the basket again trading at a premium of 1.005397 times the sum of the individual products, the standard deviation being 0.00109086). If the z-score of the deviation is sufficently high we will then buy the relatively cheap asset and sell the relatively expensive asset, hoping to make a gain if the price ratio reverts back to the mean.

This worked reasonably well, but we were not able to make up any ground, so we stayed at 48th place after this round.

</details>
<details>
<summary><h2>Round 4️⃣</h2></summary>
  
Coconuts and coconut coupons were introduced in round 4. Coconut coupons were the 10,000 strike call option on coconuts, with a time to expiry of 250 days. The price of coconuts hovered around 10,000, so this option was near-the-money. 

In this round, our approach was relatively straightforward. We used the Black-Scholes model to calculate the implied volatility of the options on coconut coupons. Once plotted, the implied volatility oscillated around a value of approximately 16%. Based on this observation, we implemented a mean-reverting strategy similar to what we used in round 3. Specifically, we calculated the delta of the coconut coupons at each point in time to hedge with coconuts, aiming to gain pure exposure to volatility.

### Theoretical Background: Black-Scholes Model

The Black-Scholes model is a mathematical model for pricing an options contract. It assumes that the price of the underlying asset follows a geometric Brownian motion with constant volatility and drift. The model is derived from the following Stochastic Differential Equation (SDE):

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

where:
- $S_t$ is the price of the underlying asset at time $t$,
- $\mu$ is the drift rate of the asset,
- $\sigma$ is the volatility of the asset,
- $dW_t$ is a Wiener process or Brownian motion.

To price an option, the Black-Scholes model uses a risk-neutral measure, where the drift rate $\mu$ is replaced by the risk-free interest rate $r$. The solution to the SDE leads to the Black-Scholes formula for a European call option price $C$:

$$C(S_t, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)$$

where:
- $C$ is the call option price,
- $S_t$ is the current price of the asset,
- $K$ is the strike price of the option,
- $T$ is the time to maturity,
- $N(\cdot)$ is the cumulative distribution function of the standard normal distribution,
- $d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$,
- $d_2 = d_1 - \sigma\sqrt{T-t}$.

### Derivation of the Black-Scholes Formula

To derive the formula, we start with the SDE for the underlying asset's price:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Applying the Itô's lemma to a function \( C(S, t) \) (representing the option price), we get:

$$dC = \frac{\partial C}{\partial S} dS + \frac{\partial C}{\partial t} dt + \frac{1}{2} \frac{\partial^2 C}{\partial S^2} dS^2$$

Substituting the SDE for \( dS \) and noting \( dS^2 = (\sigma S)^2 dt \), we obtain:

$$dC = \frac{\partial C}{\partial S} (\mu S dt + \sigma S dW) + \frac{\partial C}{\partial t} dt + \frac{1}{2} \frac{\partial^2 C}{\partial S^2} \sigma^2 S^2 dt$$

In the risk-neutral world, the expected return \( \mu \) is replaced by the risk-free rate \( r \). Therefore, under the risk-neutral measure, the equation simplifies to:

$$dC = \frac{\partial C}{\partial S} (r S dt + \sigma S dW) + \frac{\partial C}{\partial t} dt + \frac{1}{2} \frac{\partial^2 C}{\partial S^2} \sigma^2 S^2 dt$$

Eliminating the stochastic term \( dW \) by a hedging argument and equating the deterministic terms, we arrive at the Black-Scholes partial differential equation (PDE):

$$\frac{\partial C}{\partial t} + r S \frac{\partial C}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} = rC$$

Solving this PDE, with boundary conditions corresponding to the payoff of a European call option, results in the Black-Scholes formula for the call option price.

### Trading Strategy: Mean Reversion and Delta Hedging

In practice, we observed that the implied volatility (vol) of the coconut coupon options fluctuated around a mean value of approximately 16%. To exploit this mean-reverting behavior, we used a trading strategy that involved hedging our exposure to delta (the sensitivity of the option's price to changes in the price of the underlying asset) using the underlying coconuts.

The delta $\Delta$ of an option measures how much the price of the option is expected to change per unit change in the price of the underlying asset. For our strategy, the delta of the coconut coupons was approximately 0.53.

Since the coupons were far from expiry, the gamma (the rate of change of delta with respect to the price of the underlying asset) was not a significant factor.

Using the BS formula worked decently well so we found ourselves in 18th place (and then finally in place 23 after the manual score of some teams got readjusted).

![Round 4 Place](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Round%20four%20place.jpg)

</details>
<details>
<summary><h2>Round 5️⃣</h2></summary>

In round 5 all prior trade history got annotated, so we were able to reconstruct the trade parties to find alpha. We were pretty conservative, only using very clear signals. The only signal we (and other teams it turns out) was Rianna being on the money when it comes to roses, always selling at the top and buying at the bottom. Using this made us improve from 22nd to 23rd.


</details>

For the open-source tools we want to again give credit to [Jasper van Merle](https://github.com/jmerle). For this write up we followed the outline of the excellent report by the second place finish of [linear utility](https://github.com/ericcccsliu/imc-prosperity-2). Some of the ideas were the ones featured from the [Cardinals](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal/tree/main).
