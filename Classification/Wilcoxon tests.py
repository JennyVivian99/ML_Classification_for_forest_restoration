# Mann-Whitman test
# Import library
import scipy.stats as stats

# Accuracy values DT
# x is 10_R, y is 24_R
x = [1, 1, 0.87, 0.73, 0.87, 1, 0.87, 0.87, 1, 0.87]
y = [1, 1, 1, 0.73, 1, 1, 1, 1, 1, 0.53]
# Run Wilcoxon and Mann-Whitman test
#R10_R24 = stats.wilcoxon (x,y,alternative = "less")
#print("Accuracy wilcoxon DT 10R and 24R", R10_R24.pvalue)
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy mannwhitney DT R10 and R24", R10_R24.pvalue)

# Accuracy values DT
# x is G_R, y is 2_R
x = [1, 0.93, 1, 0.8, 1, 1, 1, 0.87, 0.87, 1]
y = [0.8, 0.8, 1, 0.8, 0.93, 0.87, 0.87, 0.73, 0.87, 1]
# Run Mann-Whitman test
GR_2R = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy mannwhitney DT GR and 2R", GR_2R.pvalue)

# Accuracy values AdaBoost
# x is 10_R class accuracy values, y is 24_R for the respective merged datasets
x = [0.73,0.73,1,0.93,1,1,0.87,0.87,1,1]
y = [1, 1, 0.93, 1, 1, 0.93, 1, 1, 1, 1]
# Run Wilcoxon and Mann-Whitman test
#R10_R24 = stats.wilcoxon (x,y,alternative = "less")
#print("Accuracy AdaBoost per class wilcoxon",R10_R24.pvalue)
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy AdaBoost mannwhitney R10 and R24",R10_R24.pvalue)

# Accuracy values AdaBoost
# x is G_R class accuracy values, y is 2_R for the respective merged datasets
x = [1,1,1,1,1,1,1,1,1,1]
y = [0.93, 0.87, 1, 0.87, 1, 1, 0.93, 0.93,0.87,1]
# Run Mann-Whitman test
GR_2R = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy AdaBoost mannwhitney GR and 2R",GR_2R.pvalue)

# Accuracy values DT
# y is accuracies from original dataset of DT, x are accuracies from 10-R merged dataset, always in DT. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
y = [1,1,1,1]
x = [0.87, 1, 0.87, 0.87]
# Run Mann-Whitman test
Or_R10 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy DT total original dataset-10R mannwhitney",Or_R10.pvalue)

# Accuracy values DT
# y is accuracies from original dataset of DT, x are accuracies from 10-R merged dataset, always in DT. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
y = [1,1,1,1]
x = [1,1,1,1]
# Run Mann-Whitman test
Or_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy DT total original dataset-24R mannwhitney",Or_R24.pvalue)

# Accuracy values DT
# y is accuracies from original dataset of DT, x are accuracies from GR merged dataset, always in DT. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
x = [1,1,1,0.87]
y = [1,1,1,1]
# Run Mann-Whitman test
Or_GR = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy DT total original dataset-GR mannwhitney",Or_GR.pvalue)

# Accuracy values DT
# y is accuracies from original dataset of DT, x are accuracies from 2R merged dataset, always in DT. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
x = [0.93,0.87,0.87,0.73]
y = [1,1,1,1]
# Run Mann-Whitman test
Or_2R = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy DT total original dataset-2R mannwhitney",Or_2R.pvalue)

# Accuracy values AdaBoost
# y is accuracies from original dataset of AdaBoost, y are accuracies from 10-R merged dataset. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
y = [1,1,1,1]
x = [1,1,0.87,0.87]
# Run Mann-Whitman test
Or_R10 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy AdaBoost total original dataset-10R mannwhitney",Or_R10.pvalue)

# Accuracy values AdaBoost
# x is accuracies from original dataset of AdaBoost, y are accuracies from 24R merged dataset. X values are from 0-1-2-3 seeds,
# while y values are from 4-5-6-7 seeds, to make them independent
y = [1,1,1,1]
x = [1,0.93,1,1]
# Run Mann-Whitman test
Or_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy AdaBoost total original dataset-24R mannwhitney",Or_R24.pvalue)



# Class accuracy (recall) values DT
# x is 10_R class accuracy values, y is 24_R for the respective merged datasets
x = [1,1,1,0.5,0.75,1,1,0.67,1,0.6]
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Run Mann-Whitman test
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy per class mannwhitney DT R10-R24",R10_R24.pvalue)


# Class Accuracy (recall) values AdaBoost
# x is 10_R class accuracy values, y is 24_R for the respective merged datasets
x = [0.71,0.8,1,0.75,1,1,1,1,1,1]
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Run Mann-Whitman test
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Accuracy AdaBoost per class mannwhitney",R10_R24.pvalue)


# Precision values
# Class precision values DT
# x is 10_R class precision values, y is 24_R for the respective merged datasets
x = [1,1,1,0.5,0.75,1,1,0.67,1,0.6]
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Run Mann-Whitman test
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Precision mannwhitney per class DT",R10_R24.pvalue)

# Class Precision values AdaBoost
# x is 10_R class precision values, y is 24_R for the respective merged datasets
y = [1,1,1,1,1,1,1,1,1,1]
x = [0.70, 0.57, 1, 1, 1, 1, 0.75, 0.67, 1, 1]
# Run Mann-Whitman test
R10_R24 = stats.mannwhitneyu (x,y,alternative = "less")
print("Precision mannwhitney per class AdaBoost",R10_R24.pvalue)
