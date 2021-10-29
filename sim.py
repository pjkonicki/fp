
import random
import os
from collections import defaultdict
from util import Dist

class Model(object):
	def __init__(self):
		self.incomes = dict()
		self.expenses = dict()
		self.accounts = dict()
		self.transfers = dict()

	def run(self):
		pass

	def update(self, year, month, market):
		self.year = year
		self.month = month
		for acct in self.accounts.values():
			acct.update(year, month, market)
		for inc in self.incomes.values():
			inc.update(year, month)
		for exp in self.expenses.values():
			exp.update(year, month)
		for tr in self.transfers.values():
			tr.update(year, month)

	def income(self, name, inc=None):
		if inc is None:
			return self.incomes[name]
		self.incomes[name] = inc
		inc.set_name(name)
		return inc

	def expense(self, name, exp=None):
		if exp is None:
			return self.expenses[name]
		self.expenses[name] = exp
		exp.set_name(name)
		return exp

	def account(self, name, acct=None):
		if acct is None:
			return self.accounts[name]
		self.accounts[name] = acct
		acct.set_name(name)
		return acct

	def transfer(self, name, tr=None):
		if tr is None:
			return self.transfers[name]
		self.transfers[name] = tr
		tr.set_name(name)
		return tr

	def report(self, outdir):
		if not os.path.exists(outdir):
			os.mkdir(outdir)
		for acct in self.accounts.values():
			with open(os.path.join(outdir, acct.name), 'w') as f:
				f.write(str(acct))

class Sim(object):
	def __init__(self, model, start, end, summary_every_n_years=10, ignore_accounts=['Income', 'RSUs']):
		self.model = model
		self.start = start
		self.end = end
		self.ignore_accounts = ignore_accounts
		self.summary_every_n_years = summary_every_n_years
		self.summary = dict()
		

	def fmt(self, n, width=13):
		if abs(n) < 0.001:
			n = 0
		return ('{:>%ds}' % width).format('${:,.0f}'.format(n))

	def accounts(self):
		return [acct for acct in self.model.accounts.values() if not acct.name in self.ignore_accounts]

	def balances(self):
		balances = [acct.balance() for acct in self.accounts()]
		total = sum(balances)
		return balances + [total]

	def run(self, quiet=False):
		market = Dist(0.07, 0.10)
		self.model.setup()

		headers = ''.join(['{:>13s}'.format(acct.name) for acct in self.accounts()])
		if not quiet:
			print('Year' + headers + '{:>13s}'.format('Total'))

		for year in range(self.start, self.end):
			if not quiet:
				print(('%d' % year) + ''.join([self.fmt(bal) for bal in self.balances()]))
			for month in range(1, 13):
				self.model.update(year, month, market.get_monthly())
				self.model.run()

			if year % self.summary_every_n_years == 0:
				self.summary[year] = defaultdict(int)
				total = 0
				for acct in self.accounts():
					total += acct.balance()
					if acct.category is not None:
						self.summary[year][acct.category] += acct.balance()
					self.summary[year]['Total'] = total

		if not quiet:
			print(('%d' % self.end) + ''.join([self.fmt(bal) for bal in self.balances()]))

class MC(object):
	def __init__(self, model, start, end):
		self.model = model
		self.start = start
		self.end = end

	def run_once(self):
		sim = Sim(self.model, self.start, self.end)
		sim.run()

	def run(self, n, summary_every_n_years=10):
		summary = defaultdict(lambda: defaultdict(list))
		fails = 0
		for i in range(n):
			random.seed(i)
			sim = Sim(self.model, self.start, self.end, summary_every_n_years)
			try:
				sim.run(True)
			except:
				fails += 1
			for year, stats in sim.summary.items():
				for key, val in stats.items():
					summary[year][key].append(val)

		for year, stats in summary.items():
			print('\n{:>18}  {:>13} {:>13} {:>13} {:>13} {:>13}'.format(year, '10%', '20%', '50%', '80%', 'Mean'))
			for key, vals in sorted(stats.items()):
				if len(vals) < n:
					vals = [0] * (n - len(vals)) + vals
				vals = sorted(vals)
				print('{:>18}: {} {} {} {} {}'.format(
					key, 
					sim.fmt(vals[int(n * 0.1)]),
					sim.fmt(vals[int(n * 0.2)]),
					sim.fmt(vals[int(n * 0.5)]),
					sim.fmt(vals[int(n * 0.8)]),
					sim.fmt(sum(vals) / n), 
				))
		print('\nFailure rate: {:.1f}%'.format(100 * fails / n))
