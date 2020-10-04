from sklearn import metrics
import model_evaluation as mlu

l1 = [0, 1, 1, 1, 0, 0, 0, 1]
l2 = [0, 1, 0, 1, 0, 1, 0, 0]

accuracy = mlu.accuracy(l1, l2)
print("Accuracy")
print("=================")
print(accuracy)
print("\n=================")
print("Accuracy Using sklearn")
print(metrics.accuracy_score(l1, l2))

print("\n=================")
print("Print TP, FP, TN, FN")
tp = mlu.true_positive(l1, l2)
print("True Positive : ", tp)
fp = mlu.false_positive(l1, l2)
print("False Positive : ", fp)
tn = mlu.true_negative(l1, l2)
print("True Negative : ", tn)
fn = mlu.false_negative(l1, l2)
print("False Negative : ", fn)
print("\n=================")
print("Accuracy using TP, FP, TN, FN")
accuracy = mlu.accuracy_new(l1, l2)
print("Accuracy : ", accuracy)

print("\n=================")
print("Precision Calculation")
precision = mlu.precision(l1, l2)
print("Precision : ", precision)
