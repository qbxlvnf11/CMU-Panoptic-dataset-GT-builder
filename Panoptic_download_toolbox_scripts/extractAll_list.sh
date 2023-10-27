#!/bin/bash
# Helper script to run other extraction tasks
# Input argument is output format for image files (png or jpg)

# Format for extracted images.
# Use png for best quality.
fmt=${2-jpg}

# (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
nodes=(3 6 12 13 23)
datasets=("160422_ultimatum1" "160224_haggling1" "160226_haggling1" "161202_haggling1" "160906_ian1" "160906_ian2" "160906_ian3" "160906_band1" "160906_band2" "160906_pizza1" "160422_haggling1" "160906_ian5" "160906_band4")

numVGAViews=0 #Specify the number of vga views you want to donwload. Up to 480
numHDViews=${#nodes[@]} #Specify the number of hd views you want to donwload. Up to 31

counter=0
for datasetName in ${datasets[@]}
	do

	((counter++))
	echo '---------------'
	echo $counter
	echo 'datasets: '$datasetName

	if [ $counter -ne 1 ]
	then
		cd ..
	fi
	
	cd $datasetName

	# Extract 3D Keypoints
	if [ -f hdPose3d_stage1_coco19.tar ]; then
		echo '';
		echo '-- hdPose3d_stage1_coco19 --';
		tar -xf hdPose3d_stage1_coco19.tar
	fi
	
	# Extract HD images
	../hdImgsExtractor.sh ${fmt}
done
