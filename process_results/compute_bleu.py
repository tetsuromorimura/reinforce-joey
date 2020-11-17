import json
import numpy as np 

def main():
	with open("bleu.json", "r") as file: 
		bleu_dict = json.load(file)
	for entry in bleu_dict:
		print(entry)
		for domain in bleu_dict[entry]:
			print(domain)
			for variant in bleu_dict[entry][domain]:
				print(variant)
				print("avg: ", np.mean(bleu_dict[entry][domain][variant]))
				print("std: ", np.std(bleu_dict[entry][domain][variant]))

if __name__ == '__main__':
	main()