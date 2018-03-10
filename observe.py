import pandas as pd 
import numpy as np 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import csv

SEX, INCOME, WORK, WORK_DESCRIBE, JOB, JOB_DESCRIBE, SET = [],[],[],[],[],[],[]
testx = []
def start():
	with open('train.csv') as fin:
		next(fin, None)
		reader = csv.reader(fin)
		for line in reader:
			temp = []
			SEX.append(int(line[9]))
			INCOME.append(int(line[52])) #'DL0' main income earner 1: myself 2: others
			WORK.append(int(line[53])) #'DL1' 1-10,96,99
			WORK_DESCRIBE.append(line[54]) #'DL1_ other' 
			JOB.append((np.nan) if line[55] == '' else JOB.append(float(line[55]))) #'DL2' 1-32
			JOB_DESCRIBE.append(line[56])
			for item in line[10:]:
				try:
					temp.append(int(item))
				except ValueError:
					temp.append(-1)
			SET.append(temp)

	with open('test.csv') as fin:
		next(fin, None)
		reader = csv.reader(fin)
		for line in reader:
			temp = []
			for item in line[9:]:
				try:
					temp.append(int(item))
				except ValueError:
					pass
			testx.append(temp)


def house_income_female(): 
	cnt1, cnt2, count_tot1, count_tot2 = 0, 0, 0, 0
	for i in range(len(SEX)):
		if INCOME[i] == 2 and WORK[i] == 1:
			count_tot1 += 1
			if SEX[i] == 1:
				cnt1 += 1
		if INCOME[i] == 1:
			count_tot2 += 1
			if SEX[i] ==1:
				cnt2 += 1

	print "household income from other, work = 1 is female "+str(cnt1/float(count_tot1))
	print "household income by myself is female "+str(cnt2/float(count_tot2))
	

def female_work(): #household main income -- work period
	work_myself = []
	work_myself_female = []
	work_others = []	
	work_others_female = []
	for i in range(len(SEX)):
		if INCOME[i] == 2:
			work_others.append(str(WORK[i]))
			if SEX[i] == 1:
				work_others_female.append(str(WORK[i]))
		if INCOME[i] == 1:
			work_myself.append(str(WORK[i]))
			if SEX[i] == 1:
				work_myself_female.append(str(WORK[i]))
	work_myself_count = Counter(work_myself)
	work_myself_count_female = Counter(work_myself_female)
	work_others_count = Counter(work_others)
	work_others_count_female = Counter(work_others_female)


	values = work_myself_count.values()
	keys = work_myself_count.keys()
	bar_x_locations = np.arange(len(keys))
	plt.bar(bar_x_locations-0.15, values, width = 0.3, align = 'center', color = 'b')
	plt.xticks(bar_x_locations, keys)
	values = work_myself_count_female.values()
	keys = work_myself_count_female.keys()
	plt.bar(bar_x_locations+0.15, values, width = 0.3, align = 'center', color = 'r')
	plt.xlabel("household income by myself(women)work type histgram")
	plt.ylabel("number")
	plt.savefig('household income by myself.png')
	plt.close()

	values1 = work_others_count.values()
	keys = work_others_count.keys()
	bar_x_locations = np.arange(len(keys))
	plt.bar(bar_x_locations-0.15, values1, width = 0.3, align = 'center', color = 'b')
	plt.xticks(bar_x_locations, keys)
	values2 = work_others_count_female.values()
	keys = work_others_count_female.keys()
	plt.bar(bar_x_locations+0.15, values2, width = 0.3, align = 'center', color = 'r')
	for i in range(len(values1)):
		print float(values2[i])/values1[i]
	plt.xlabel("household income by others(women)work type histgram")
	plt.ylabel("number")
	plt.savefig('household income by others.png')
	plt.close()


if __name__ == '__main__':
	house_income_female()
	female_work()
	start()

	



		


