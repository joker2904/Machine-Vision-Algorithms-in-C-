for f in $(find . -name "*.png" | sort); do echo $f >> filenames.txt; done
