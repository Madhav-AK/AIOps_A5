from pprint import pprint
from partB.eta_mlflow import promotion_gate

print("Default thresholds")
pprint(promotion_gate())

print("\nmin_rel_gain=0.60")
pprint(promotion_gate(min_rel_gain=0.60))
