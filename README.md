# Landscape or Cityscape

The goal of this project is to train a convolutional neural net to classify images as either a picture of a landscape or a picture of a city. To gather data for this project I scraped about 5500 images from the subreddits [r/EarthPorn](https://www.reddit.com/r/EarthPorn/) and [r/CityPorn](https://www.reddit.com/r/CityPorn/). The script I used to get this data can be found in src/gather_subreddit.py and can be called from the command line. 

These subreddits are well moderated collections of landscape and city pictures respectively. Because these subreddits are well moderated you can be fairly certain that the pictures in each subreddit are not mislabeled, especially if they have at least 100 upvotes. Because of this I decided to use this data for an image classifier to determine if a picture is of a natural landscape or an urban landscape. The code for my process can be found in [this notebook](https://github.com/GougeC/Landscape_or_Cityscape/blob/master/Landscape_Or_Cityscape.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/GougeC/Landscape_or_Cityscape/blob/master/Landscape_Or_Cityscape.ipynb)), but my general process was to use a pretrained CNN (in this case [VGG16](https://arxiv.org/abs/1409.1556)) as feature extraction and replace the top layers of the network with new layers. For this project I trained a variety of top networks and found the performance of both a single dense layer and two dense layers to be very similar (about 95% accuracy). 

![city picture](https://github.com/GougeC/Landscape_or_Cityscape/blob/master/src/cityexample.jpg?raw=true)
![earth picture](https://github.com/GougeC/Landscape_or_Cityscape/blob/master/src/earthexample.jpg?raw=true)

