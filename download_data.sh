echo "This script downloads datasets or camera stats files used in the Campari project."
echo "Choose from the following options:"
echo "0 - Cats Dataset"
echo "1 - CelebA Dataset"
echo "2 - Carla Dataset"
echo "3 - Chairs1 Dataset"
echo "4 - Chairs2 Dataset"
echo "5 - Camera stats files"
read -p "Enter dataset ID you want to download: " ds_id

if [ $ds_id == 0 ]
then
    echo "You chose 0: Cats Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/campari/data/cats.zip
    echo "done! Start unzipping ..."
    unzip cats.zip
    echo "done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 1: CelebA Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/campari/data/celeba.zip
    echo "done! Start unzipping ..."
    unzip celeba.zip
    echo "done!"
elif [ $ds_id == 2 ]
then
    echo "You chose 2: Carla Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip
    echo "done! Start unzipping ..."
    unzip carla.zip -d carla
    echo "done!"
elif [ $ds_id == 3 ]
then
    echo "You chose 3: Chairs1 Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/campari/data/chairs1.zip
    echo "done! Start unzipping ..."
    unzip chairs1.zip
    echo "done!"
elif [ $ds_id == 4 ]
then
    echo "You chose 4: Chairs2 Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/campari/data/chairs2.zip
    echo "done! Start unzipping ..."
    unzip chairs2.zip
    echo "done!"
elif [ $ds_id == 5 ]
then
    echo "You chose 5: Camera Stats Files"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/campari/stats_files/stats_files.zip
    echo "done! Start unzipping ..."
    unzip stats_files.zip
    echo "done!"
else
    echo "You entered an invalid ID!"
fi
