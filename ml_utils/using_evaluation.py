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

print("\n=================")
print("Recall Calculation")
recall = mlu.recall(l1, l2)
print("Recall : ", recall)

print("\n=================")
print("F1 Score Calculation")
f1_score = mlu.f1_score(l1, l2)
print("F1 Score : ", f1_score)

print("\n=================")
print("True Positive Rate")
tpr = mlu.true_positive_rate(l1, l2)
print("TPR :", tpr)
print("\n=================")
print("False Positive Rate")
fpr = mlu.false_positive_rate(l1, l2)
print("FPR :", fpr)
