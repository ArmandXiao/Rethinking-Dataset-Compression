#!/bin/bash

wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

echo "Downloaded and extracted validation data."
echo "Preprocessed data is in val/ directory."

export IMAGENET_VAL_DIR=$(pwd)/val/

echo "Set IMAGENET_VAL_DIR environment variable to $(pwd)/val/."