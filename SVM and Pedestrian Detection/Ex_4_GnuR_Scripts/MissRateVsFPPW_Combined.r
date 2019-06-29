# Name for Saving
imagename <- paste("MissRateVsFPPW_Combined.pdf")

# Read Data
dataPosRaw <- read.table("data/SvmPredictTestPos.mat", header=F)
dataNegRaw<- read.table("data/SvmPredictTestNeg.mat", header=F)

# Loop over all SVM outputs
for (k in 1:3)
{
	dataPos = dataPosRaw[,1]
	dataNeg = dataNegRaw[,1]

	lenPos <- length(dataPos)
	lenNeg <- length(dataNeg)

	# Create a sequence of 20 thresholds
	minThresh = min(dataPos)
	maxThresh = max(dataPos)
	myThreshs <- seq(minThresh, maxThresh,length.out=20)
	numSteps <- length(myThreshs)

	# Create container "dataFrame" to store the calculated values
	dataFrame <- data.frame( missRate = numeric(), FPPW = numeric(),Thresh=numeric(), FN=numeric())


	# Loop through all threshold values, calc missRate and FPPW
	for( i in 1:numSteps)
	{
		numFalseNegatives <-  length(which(dataPos > myThreshs[i]))
		numFalsePositives <-  length(which(dataNeg < myThreshs[i]))

		missRate <- numFalseNegatives / length(dataPos)
		FPPW <- numFalsePositives / (length(dataNeg)+length(dataPos))

		print(FPPW)

		dataFrame[i,] <- c( missRate, FPPW, myThreshs[i], numFalseNegatives )
	}
	print(dataFrame)
	
	# Plot results
	if(k==1){
		plot((dataFrame$FPPW),(dataFrame$missRate), col="darkblue",type="o",lty=1,lwd=3, pch=25, cex=0.5, main="DET Curve",xlab="false positives per window (FPPW)", ylab="miss rate",log="xy") 
	}
	if(k==2){
		lines((dataFrame$FPPW),(dataFrame$missRate), col="darkred",type="o",lty=2,lwd=3, pch=25, cex=0.5)
	}
	if(k==3){
		lines((dataFrame$FPPW),(dataFrame$missRate), col="darkorange3",type="o",lty=3,lwd=3, pch=25, cex=0.5) 
	}
}
# Legend
col1 = "darkblue"
col2 = "darkred"
col3 = "darkorange3"
legend(0.05,0.5, c("C = 0.01","C = 100"),col=c(col1,col3),lty=c(1,1),lwd=c(2.5,2.5))

# Save results
savePlot(file=imagename, type="pdf")
