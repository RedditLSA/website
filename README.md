# Reddit LSA Website
This repository is the code for my [webpage](https://www.redditlsa.com) which is inspired by [this FiveThirtyEight article](https://fivethirtyeight.com/features/dissecting-trumps-most-rabid-online-following/). The benefit of this website over the original article is the interactive map view which uses a more sophisticated algorithm than the original article. 
## Architecture
All of the data is in the `redditlsa/data` folder. This contains four objects
- `X.npz`: A sparse matrix with 40,875 rows and 1,800 columns representing the subreddit vectors
- `subs_by_popularity.pkl`: A list of subreddits sorted in descending order of the number of authors
- `sub_to_index.pkl`: A dictionary that converts subreddits to their corresponding row index in the X matrix
- `index_to_sub.pkl`: The inverse of sub_to_index
The data is created by code that doesn't exist in this repository because this is just for the webpage.
## Building
This is a regular Django project, with no dependencies that can't be installed outside of pip, so this should work:
```bash
pip install -r requirements.txt
python manage.py runserver
```
