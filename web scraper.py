import requests
from bs4 import BeautifulSoup

def web_scraper(url, html_class):
    # inputs 2 strings, one a url, and the other the class of which the body
    # of the text we are interested in is found
    # returns a list of integers that are ASCII representation of the characters

    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    # Find the class
    s = soup.find('div', class_=html_class)
    lines = s.find_all('p')

    # Extract the text
    txt = ""
    for line in lines:
        txt = txt + line.text

    # Split into characters
    chars = [char for char in txt]

    # Map it to integers as we have for the model
    output_tokens = list(map(ord, chars))
    print(output_tokens)
    return output_tokens

# an exmaple using a Reuters article
web_scraper('https://www.reuters.com/markets/europe/cryptos-latest-meltdown-leaves-punters-bruised-bewildered-2022-06-21/',
            'article-body__content__17Yit paywall-article')

# The web scraper should be ready to work in conjunction with the model to identify the names, organizations and locations
# from any article on the internet.