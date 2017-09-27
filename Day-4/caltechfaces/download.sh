wget http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/WebFaces_GroundThruth.txt -O annotations.txt
wget http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/Caltech_WebFaces.tar -O images.tar
tar -xf images.tar
mkdir -p images
mv *.jpg images/
rm images.tar