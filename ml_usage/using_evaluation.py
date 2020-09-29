from sklearn import metrics
import ml_utils.mlutils as mlu

l1 = [0, 1, 1, 1, 0, 0, 0, 1]
l2 = [0, 1, 0, 1, 0, 1, 0, 0]

accuracy = mlu.accuracy(l1, l2)
print(accuracy)

print("Using sklearn")
metrics.accuracy_score(l1, l2)
