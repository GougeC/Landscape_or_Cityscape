import praw
import requests
import os
from PIL import Image
import io
import time
import datetime
import numpy as np
import sys
from scipy.misc import imread

def get_and_save_sub_images(subreddit, date, min_votes):
    '''saves images from the given subreddit since date (d/m/y) with >= min_votes in a folder named subreddit_[day]_month'''
    folders = []
    urls = get_urls(subreddit, date, min_votes)
    today = datetime.datetime.today()
    month,day = today.month, today.day
    if not os.path.exists("images_{}_{}".format(month,day)):
        os.mkdir("images_{}_{}".format(month,day))
    for fol in ['train','test','validation']:
        if not os.path.exists("images_{}_{}/{}".format(month,day,fol)):
            os.mkdir("images_{}_{}/{}".format(month,day,fol))
    for fol in ['train','test','validation']:
        folder = "images_{}_{}/{}/{}".format(month, day,fol,subreddit)
        folders.append(folder)
        os.mkdir(folder)
    folder+='/'
    print("saving {} images to {} ".format(len(urls),folder))
    i = 0
    for ind, url in enumerate(urls):

        if i%10 == 9:
            fol = 'test'
        elif i%10 == 8:
            fol = 'validation'
        else:
            fol = 'train'

        folder = "images_{}_{}/{}/{}/".format(month, day,fol,subreddit)
        try:
            response = requests.get(url)
            if response.content:
                img = Image.open(io.BytesIO(response.content))
                name = "image{}{}".format(ind,url[-4:])
                img.save(folder+name)
                i+=1
        except:
            continue
        if i%100 == 0:
            print("saved {} images from r/{}".format(i,subreddit))
    print("done with /r/{}".format(subreddit))
    return folders

def get_urls(subreddit,date,min_votes):
    ''' returns a list of urls of images posted on the sub since date that have >= min_votes upvotes'''
    print("getting urls from r/{}".format(subreddit))
    current_time = time.time()
    start_time = time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y").timetuple())
    urls = []
    reddit = praw.Reddit(client_secret = os.environ['REDDIT_API_SECRET'],
                     client_id = os.environ['REDDIT_API_ID'] ,
                     user_agent = 'gathering data script by /u/GougeC')
    i = 0
    for post in reddit.subreddit(subreddit).submissions(start_time,current_time):
        if post.ups >= min_votes:
            url = post.url
            if url[-4:] in ('.png' ,'.jpg'):
                urls.append(url)
                i+=1
                if i%100 ==0:
                    print("got {} urls".format(i))
    return urls

def get_multiple_subreddits(subreddit_dict):
    """ Takes in a dictionary of subreddits in the form of
        {subreddit : (date , minimum_upvotes)
        where the date is in the form d/m/y and is the earliest posts you want to pull
        and minimum_upvotes is the number of upvotes that a post most have to be included
    """
    folders = []
    for sub, info in subreddit_dict.items():
        fol = get_and_save_sub_images(sub, info[0], info[1])
        folders.extend(fol)

def eliminate_broken_images(folders):
    imgs = []
    for folder in folders:
        paths = os.listdir(folder)
        paths = [folder+'/'+i for i in paths]
        imgs.extend(paths)
    broken = []
    for img in imgs:
        try :
            imread(img)
        except:
            broken.append(img)
    for f in broken:
        os.remove(f)


if __name__ == '__main__':
    """
    This allows this script to be called from the command line and get the images specified
    by the parameters put in.
    EX:
    python gather_subreddit.py earthporn 1/1/2017 100

    would save every direct link image from r/earthporn that had over 100 upvotes.
    """
    subreddit,date,minimum_upvotes = sys.argv[1],sys.argv[2],int(sys.argv[3])
    folders = get_and_save_sub_images(subreddit,date,minimum_upvotes)
    eliminate_broken_images(folders)
    print("finished.")
