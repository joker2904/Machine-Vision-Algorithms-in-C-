all: nr1 nr2 nr3

nr1:
	g++ nr1.cc   -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -o faceDetector
nr2:
	g++ nr2.cc   AdaBoost.cc WeakClassifier.cc -o adaBoost

nr3:
	g++ nr3.cc   AdaBoost.cc WeakClassifier.cc -lopencv_core -lopencv_highgui -lopencv_imgproc -o tracking
