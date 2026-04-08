# NMDS for ML paper
# library
library(vegan)
library(ggplot2)

# Load the data
data_ml<-read.table("Reviewed_Holistic_Dataset_NMDS.csv", sep=",", h=T)
# remove column for sample ID
data_ml<-data_ml[,-1]
# Check
summary(data_ml)
# make landcover as factor
data_ml$Landcover<-as.factor(data_ml$Landcover)
# order the dataframe
order<-c("Grassland","2 years old", "10 years old", "24 years old", "Remnant")
data_ml$Landcover<-factor(data_ml$Landcover, levels= order)
data_ml <- data_ml[order(data_ml$Landcover), ]

# Conduct NMDS analysis
# check
class(data_ml$Landcover)
data_ml$Landcover
# Check
dim(data_ml)
structure(data_ml)
data_ml_NMDS<-as.matrix(data_ml[,2:185])
NMDS<-metaMDS(data_ml_NMDS, distance = "jaccard")

# Extract the NMDS coordinates (scores)
data_scores <- as.data.frame(scores(NMDS)$sites)

# Add Landcover column to the coordinates and ensure data_scores and Landcover have the same number of rows
nrow(data_scores)
nrow(data_ml)
data_scores$Landcover <- data_ml$Landcover

# Create the plot
# 1. Create a color palette based on Landcover levels
colors <- c("red", "pink", "lightblue", "orange", "darkgreen")

ggplot(data_scores, aes(x = NMDS1, y = NMDS2, color = Landcover, fill = Landcover)) + 
  geom_point(size = 3, alpha = 0.8) + 
  stat_ellipse(geom = "polygon", 
               alpha = 0.08, 
               level = 0.95, 
               type = "norm", 
               linewidth = 1) +
  scale_color_manual(values = colors) +
  scale_fill_manual(values = colors) +
  theme_bw() +
  labs(title = "NMDS Ordination by Landcover Type")
