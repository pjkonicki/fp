#!/usr/bin/env python3

import random
import yfinance as yf
import numpy as np
import pandas as pd
from typing import NamedTuple

M = 1000*1000
DIVIDEND_TAX_RATE=0.45
HISTORY_START = 2008
HISTORY_END = 2021
HISTORY_YEARS = HISTORY_END - HISTORY_START

class Symbol(object):
	def __init__(self, sym, amt=0):
		self.sym = sym
		self.amt = amt
		self.ticker = yf.Ticker(sym)
		self.hist = self.ticker.history(period='max', auto_adjust=False)
		returns = []
		dividends = []
		for year in range(HISTORY_START, HISTORY_END):
			for month in range(1, 13):
				next_year = year
				next_month = month + 1
				if next_month == 13:
					next_year += 1
					next_month = 1
				start = '{}-{:02d}-01'.format(year, month)
				end = '{}-{:02d}-01'.format(next_year, next_month)
				df = self.hist[start:end]
				prices = df['Close']
				try:
					returns.append(prices[-1] / prices[0])
					dividends.append(df['Dividends'].sum() / prices[-1])
				except IndexError as e:
					raise Exception("No price available for {} on {}/{}. Oldest available is {}.".format(sym, month, year, self.hist.index.min()))
		self.returns = np.array(returns)
		self.dividends = np.array(dividends)

class Portfolio(object):
	def __init__(self, syms, cash=0, taxed_account=False, n=1):
		self.symbols = [Symbol(sym) for sym in syms]
		bal = { sym: np.zeros(n) for sym in syms }
		bal['cash'] = np.full(n, cash)
		self.balance = pd.DataFrame(data=bal , dtype=np.float32)
		self.taxed_account = taxed_account
		self.n = n

	def reset(self, cash):
		self.balance[:] = 0
		self.balance['cash'][:] = cash

	def total(self):
		return self.balance.sum(axis=1)

	def update(self, dates):
		tax_rate = DIVIDEND_TAX_RATE if self.taxed_account else 0
		for sym in self.symbols:
			self.balance[sym.sym] *= sym.returns[dates]
			self.balance['cash'] += sym.dividends[dates] * self.balance[sym.sym] * (1 - tax_rate)

	def contribute(self, amt):
		self.balance['cash'] += amt

	def rebalance(self, target):
		tot = self.total()

		tgt = { sym: np.full(self.n, target[sym]) for sym in target }
		tgt['cash'] = np.zeros(self.n)
		target_df = pd.DataFrame(tgt)

		if self.taxed_account:
			# For taxed accounts, only use dividends and contributions to rebalance (don't sell)
			expected = target_df.mul(tot, axis=0)
			diffs = expected - self.balance
			diffs *= diffs > 0
			pct = self.balance['cash'] / diffs.sum(axis=1)
			self.balance += diffs.mul(pct, axis=0)
			self.balance['cash'][:] = 0

		else:
			# For untaxed accounts, sell to rebalance if necessary
			self.balance = target_df.mul(tot, axis=0)

	def __repr__(self):
		tot = self.total().mean()
		rows = [(sym.sym, self.balance[sym.sym].mean()) for sym in self.symbols]
		rows += [('cash', self.balance['cash'].mean()), ('Total', tot)]
		return '\n'.join(["{:>5s} {:>6s} {:>13s}".format(
			sym,
			'{:.1f}%'.format(amt * 100 / tot),
			'${:,.0f}'.format(amt)) for sym, amt in rows])


class Simulator(object):
	def __init__(self, portfolio):
		self.portfolio = portfolio

	def run(self, strategy, init, years, quiet=False):
		self.portfolio.reset(init)
		for year in range(years):
			for quarter in range(4):
				# Pick a random quarter from history to use to update prices
				qtr = np.random.randint(HISTORY_YEARS * 12 - 3, size=self.portfolio.n)
				for month in range(3):
					dt = (year * 12 + quarter * 3 + month) / (years * 12)
					self.portfolio.contribute(strategy.contribution(dt))
					self.portfolio.rebalance(strategy.target(dt))
					self.portfolio.update(qtr + month)
			if not quiet:
				print(year)
				print(self.portfolio)


class Strategy(object):
	def __init__(self, targets, cont):
		total = sum(targets.values())
		self.targets = { sym: targets[sym] / total for sym in targets }
		self.cont = cont
	def params(self):
		return self.targets.keys()
	def target(self, dt):
		return self.targets
	def contribution(self, dt):
		return self.cont
	def with_gradient(self, gradient, step_size=1):
		new_targets = { t: self.targets[t] * (1 + gradient.get(t, 0) * step_size) for t in self.targets }
		return Strategy(new_targets, self.cont)
	def randomize(self, factor):
		new_targets = { sym: self.targets[sym] * random.uniform(1-factor, 1+factor) for sym in self.targets }
		return Strategy(new_targets, self.cont)
	def __repr__(self):
		return '\n'.join(["{:>5s} {:>6s}".format(
			sym,
			'{:.1f}%'.format(self.targets[sym] * 100)) for sym in self.targets])


class EqualStrategy(Strategy):
	def __init__(self, portfolio, contributions):
		super().__init__({ sym.sym: 1 for sym in portfolio.symbols }, contributions)


class InterpolatingStrategy(Strategy):
	def __init__(self, s1, s2):
		self.s1 = s1
		self.s2 = s2
	def params(self):
		return [(1, k) for k in self.s1.params()] + [(2, k) for k in self.s2.params()]
	def target(self, dt):
		t1 = self.s1.target(dt)
		t2 = self.s2.target(dt)
		return { sym: t1[sym] * (1 - dt) + t2[sym] * dt for sym in t1 }
	def contribution(self, dt):
		return self.s1.contribution(dt) * (1 - dt) + self.s2.contribution(dt) * dt
	def with_gradient(self, gradient, step_size=1):
		g1 = { k: gradient[(i, k)] for i, k in gradient if i == 1 }
		g2 = { k: gradient[(i, k)] for i, k in gradient if i == 2 }
		return InterpolatingStrategy(self.s1.with_gradient(g1, step_size*2), self.s2.with_gradient(g2, step_size*2))
	def randomize(self, factor):
		return InterpolatingStrategy(self.s1.randomize(factor), self.s2.randomize(factor))
	def __repr__(self):
		t1 = self.target(0)
		t2 = self.target(1)
		return '\n'.join(["{:>5s} {:>6s} -> {:>5s}".format(
			sym,
			'{:.1f}%'.format(t1[sym] * 100),
			'{:.1f}%'.format(t2[sym] * 100)) for sym in t1])


class Optimizer(object):
	def __init__(self, portfolio, init, goal, years):
		self.portfolio = portfolio
		self.init = init
		self.goal = goal
		self.years = years
		self.sim = Simulator(portfolio)

	# strategy: strategy to optimize
	# step_size: update step size
	# delta: gradient test step size
	# epsilon: stopping condition
	def optimize(self, strategy, step_size, delta, epsilon, randomize_factor=0.0):
		success_rate = 0

		i = 0
		while True:
			print(strategy)
			old_success_rate = success_rate
			success_rate = self.trial(strategy)
			print('Success rate: {:.1f}%\n'.format(success_rate * 100))

			if abs(success_rate - old_success_rate) < epsilon:
				return strategy

			gradient = dict()
			for param in strategy.params():
				test_strategy = strategy.with_gradient({param: delta})
				test_success_rate = self.trial(test_strategy)
				gradient[param] = (test_success_rate - success_rate) / delta
				print('{} {:+.1f}%'.format(param, gradient[param] * 100))

			strategy = strategy.with_gradient(gradient, step_size)

			i += 1
			if i % 4 == 0:
				strategy = strategy.randomize(randomize_factor)
				randomize_factor *= 0.95

			step_size *= 0.95


		return strategy

	def cross_validate(self, strategy):
		success_rate = self.trial(strategy, seed=4)
		print('Cross-validate: {:.1f}%\n'.format(success_rate * 100))

	def trial(self, strategy, seed=17):
		np.random.seed(seed)
		self.sim.run(strategy, self.init, self.years, quiet=True)
		success = (self.portfolio.total() > self.goal).sum()
		return success / self.portfolio.n


# Standard ETFs used by WealthFront
wf_tickers = [
	'VTI',  # US Stocks
	'VEA',  # Foreign Stocks
	'VWO',  # Emerging Markets
	'XLE',  # Natural Resources
	'VNQ',  # Real Estate
	'TIP',  # Inflation-protected bonds
	'TFI',  # Muni Bonds
	'VIG'   # Dividend Stocks
]

# ETFs with history back to at least 2004
hist_tickers = [
	'VTI',  # US Stocks
	'IOO',  # Foreign Stocks
	'EEM',  # Emerging Markets
	'XLE',  # Natural Resources
	'IYR',  # Real Estate
	'TIP',  # Inflation-protected bonds
	'LQD',  # Corporate Bonds
	'DVY'   # Dividend Stocks
]

min_tickers = ['VTI', 'LQD']

tickers = wf_tickers

def run_test():
	p = Portfolio(['VTI', 'SPY'], n=3, taxed_account=True)
	p.contribute(1000)
	p.rebalance({'SPY': 0.4, 'VTI':0.6})
	p.update([3, 5, 7])
	print(p)

def run_sim():
	portfolio = Portfolio(tickers, n=1, taxed_account=False)
	strategy = EqualStrategy(portfolio, contributions=5000)
	sim = Simulator(portfolio)
	sim.run(strategy, init=1*M, years=10)

def run_opt():
	portfolio = Portfolio(tickers, n=10000, taxed_account=True)
	opt = Optimizer(portfolio, init=1*M, goal=2.2*M, years=10)

	# Optimize a strategy with a fixed allocation for the entire period.
	#strategy = EqualStrategy(portfolio, contributions=5000)
	#strategy = opt.optimize(strategy, step_size=15, delta=0.05, epsilon=0.001, randomize=0)

	# Optimize a strategy that moves from one allocation to another linearly over time.
	strategy = InterpolatingStrategy(
		EqualStrategy(portfolio, contributions=4000),
		EqualStrategy(portfolio, contributions=8000))
	strategy = opt.optimize(strategy, step_size=4, delta=0.1, epsilon=0.001, randomize_factor=0.2)
	opt.cross_validate(strategy)


if __name__ == '__main__':
	run_opt()

	print('Done')

