# Name for Saving
imagename <- paste("PrecisionVsRecall_Combined.pdf")

# Read Data
dataPosRaw <- read.table("data/SvmPredictTestPos.mat", header=F)
dataNegRaw<- read.table("data/SvmPredictTestNeg.mat", header=F)

# Loop over all SVM outputs
for (k in 1:3)
{
	dataPos = dataPosRaw[,k]
	dataNeg = dataNegRaw[,k]

	lenPos <- length(dataPos)
	lenNeg <- length(dataNeg)

	# Create a sequence of 20 thresholds
	minThresh = min(dataPos)
	maxThresh = max(dataPos)
	myThreshs <- seq(minThresh, maxThresh,length.out=20)
	numSteps <- length(myThreshs)

	# Create container "dataFrame" to store the calculated values
	dataFrame <- data.frame( Precision = numeric(), Recall = numeric(),Thresh=numeric(), TP=numeric(),FN=numeric(),FP=numeric())


	# Loop through all threshold values, calc missRate and FPPW
	for( i in 1:numSteps)
	{
		TruePositives  <-  length(which(dataPos < myThreshs[i]))
		FalseNegatives <-  length(which(dataPos > myThreshs[i]))
		FalsePositives <-  length(which(dataNeg < myThreshs[i]))
		
		Precision <- TruePositives / (FalsePositives+TruePositives)
		Recall    <- TruePositives / (length(dataPos))
	#	missRate <- FalseNegatives / length(dataPos)
	#	FPPW <- FalsePositives / (length(dataNeg)+length(dataPos))

		dataFrame[i,] <- c(Recall,Precision, myThreshs[i], TruePositives, FalseNegatives,FalsePositives )

	}
	print(dataFrame)

	# Plot results -> For explanation see http://de.wikibooks.org/wiki/GNU_R:_plot
	if(k==1){
		plot((dataFrame$Recall),(dataFrame$Precision),col="darkblue",type="o",lty=1,lwd=3, pch=25, cex=0.5, main="Precision vs Recall Curve",xlab="Recall", ylab="Precision") 
	}
	if(k==2){
		# lines((dataFrame$Recall),(dataFrame$Precision), col="darkred",type="o",lty=2,lwd=3, pch=25, cex=0.5)
	}
	if(k==3){
		lines((dataFrame$Recall),(dataFrame$Precision), col="darkorange3",type="o",lty=3,lwd=3, pch=25, cex=0.5) 
	}
}
# Legend
col1 = "darkblue"
col2 = "darkred"
col3 = "darkorange3"
legend(0,0.2 , c("C = 0.01","C = 100"),col=c(col1,col3),lty=c(1,1),lwd=c(2.5,2.5))

# Save results in a pdf
savePlot(file=imagename, type="pdf")
