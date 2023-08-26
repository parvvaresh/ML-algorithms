import pandas as pd
import numpy as np
import csv

class load_data:
	def __init__(self):
		pass

	def read_csv_loader(self, path_file):
		with open(path_file, "r") as f:
		  data = list(csv.reader(f, delimiter=","))

		data = np.array(data, dtype=np.float32)
		return data

	def load_numpy_text(self, path_file):
		data = np.loadtxt(path_file, delimiter=",", dtype=np.float32) 
		return data

	def load_numpy_genfromtxt(self, path_file)
		data = np.genfromtxt(path_file, delimiter=",", dtype=np.float32)
		return data

	def load_pandas(self, path_file):
		df = pd.read_csv(path_file, header=None, skiprows=0, dtype=np.float32)
		df = df.fillna(0.0)
		data = df.to_numpy()
		return data
