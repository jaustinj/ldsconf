import calendar
import functools
import logging
from itertools import product
import os
import pickle
from time import sleep

from bs4 import BeautifulSoup
import pandas as pd
import requests


def set_logging(directory, filename, level=logging.DEBUG):
    logger = logging.getLogger('LDS Crawler')
    logger.setLevel(level)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    path = directory + '/' + filename
    if not os.path.isfile(path):
        with open(path, 'w') as file:
            pass
    
    
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger

logger = set_logging('./temp', 'ldscrawler.log')
    

def exceptor(calling_level=logger.debug,
             success_level=logger.debug, 
             fail_level=logger.error,
             no_arg=False,
             sep=False):
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if sep:
                calling_level(sep)

            if no_arg:
                arg_str = 'Too Large'
                kwarg_str = ''
            else:
                arg_str = ', '.join([str(arg) for arg in args])
                kwarg_str = ', '.join(['{}={}'.format(kwd, arg) for kwd, arg in kwargs.items()])
            
            calling_level('Calling {}({}, {})'.format(func.__name__, arg_str, kwarg_str))

            try:
                response = func(*args, **kwargs)
                success_level('Call to {} Successful'.format(func.__name__))
                return response
            except Exception as e:
                fail_level('Call to {} Unsuccessful:'.format(func.__name__))
                fail_level('\t{}({})'.format(e.__class__.__name__, e))
                return 
        return wrapper
    return real_decorator
       

@exceptor()
def create_file_if_not_exists(directory, filename=None):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if filename:
        path = directory + '/' + filename
        if not os.path.isfile(path):
            with open(path, 'w') as file:
                pass
        return path
     
@exceptor(no_arg=True)   
def save_to_file(directory, filename, df):
    path = create_file_if_not_exists(directory, filename)
    df.to_csv(path)
  
    
@exceptor(no_arg=True)
def pickle_save(obj, filename):
    with open('./temp/pickles/{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
@exceptor()     
def pickle_load(directory, filename):
    with open('./temp/pickles/{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)
    
@exceptor()
def pickle_wipe():
    files = os.listdir('./temp/pickles')
    for file in files:
        os.remove('./temp/pickles/' + file)
    
    os.rmdir('./temp/pickles')

            
class LdsConfScraper(object):
    def __init__(self, months=['04', '10'], years=range(1971, 2019), languages=['eng'], delay=2):
        self.months = months
        self.years = list(years)
        self.lang = languages
        self.conferences = []
        
        self._main(delay=delay)
    
    def __str__(self):
        return 'LdsConfScraper({}-{}, {})'.format(self.years[0], self.years[-1], ', '.join(self.lang))
    
    def __repr__(self):
        return self.__str__()
    
    @exceptor(sep='\n\n-----------STARTING NEW SCRAPE----------\n')
    def _main(self, delay):
        # Create temp pickle folder for temp files
        create_file_if_not_exists('./temp/pickles', filename=None)
        for year, month, lang in product(self.years, self.months, self.lang):
            self.conferences.append(ConferencePage(year, month, lang, delay=delay))
            sleep(delay)
            
    def to_df(self):
        talks_nested = [talk for talk in [conf.talks for conf in self.conferences]]
        talks = [item for sublist in talks_nested for item in sublist]
        talk_dicts = [talk.to_dict() for talk in talks]
        return pd.DataFrame.from_records(talk_dicts)
        
        
class ConferencePage(object):
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    CONFERENCE_BASE_URL = 'https://www.lds.org/general-conference/{year}/{month}?lang={lang}'
    
    def __init__(self, year, month, lang, delay=2):
        self.year = year
        self.month = month
        self.lang = lang
        self.delay = delay
        self.url = self.CONFERENCE_BASE_URL.format(year=year, month=month, lang=lang)
        self.soup = None
        self.talk_urls = None
        self.talks = None
        
        self._main()
       
    def __str__(self):
        return 'ConferencePage({}-{}, {})'.format(self.year, self.month, self.lang)
    
    def __repr__(self):
        return self.__str__()
    
    @exceptor()
    def _get_conference_page_soup(self):
        return BeautifulSoup(requests.get(self.url, headers=self.HEADERS).content, features='html.parser')
        
    @exceptor()
    def _get_talk_url_endings(self):
        talks_layout = self.soup.find('div', {'class': 'section-wrapper lumen-layout lumen-layout--landing-3'})
        tags = talks_layout.findAll('a')
        hrefs = [t['href'] for t in tags]
        return [href for href in hrefs if href.startswith('/general-conference')]
    
    @exceptor()
    def _get_talk(self, url):
        return Talk(url, self.year, self.month, self.lang)
    
    @exceptor()
    def _get_talks(self):
        talks = []
        for url in self.talk_urls:
            talk = self._get_talk(url)
            pickle_save(talk.to_dict(), talk.title.replace(' ', '_').replace(',',''))
            talks.append(talk)
            sleep(self.delay)
            
        return talks
       
    @exceptor(sep='\n\n----------STARTING NEW CONFERENCE SCRAPE----------\n')
    def _main(self):
        self.soup = self._get_conference_page_soup()
        self.talk_urls = self._get_talk_url_endings()
        self.talks = self._get_talks()
            
        

class Talk(object):
    TALK_BASE_URL = 'https://www.lds.org{ending}'
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    def __init__(self, url_ending, year, month, lang):
        self.talk_url = self.TALK_BASE_URL.format(ending=url_ending)
        self.conf = '{} {}'.format(calendar.month_name[int(month)], year)
        self.lang = lang
        self.soup = None
        self.title = None
        self.author = None
        self.author_calling = None
        self.text = None
    
        self._main()
    
    def __str__(self):
        if self.title:
            return 'Talk(Title={}, Author={})'.format(self.title, self.author)
        else:
            return 'Talk(PENDING: url={})'.format(self.talk_url)
    
    def __repr__(self):
        return self.__str__()
    
    @exceptor()
    def _get_soup(self):
        content = requests.get(self.talk_url, headers=self.HEADERS).content
        return BeautifulSoup(content, features='html.parser')
    
    @exceptor()
    def _get_title(self):
        title_clutter = self.soup.find('h1').text
        title = title_clutter.replace('‚Äú', '').replace('‚Ä', '').replace('¶','')
        return title
    
    @exceptor()
    def _get_author(self):
        try:   
            author_clutter = self.soup.find('a', {'class': 'article-author__name'}).text
        except:
            author_clutter = self.soup.find('div', {'class': 'article-author'}).text
            
        author = author_clutter.replace('\xa0', ' ').replace('By ', '').strip()
        return author.split('\n')[0]
    
    @exceptor()
    def _get_author_calling(self):
        calling =  self.soup.find('p', {'class': 'article-author__title'}).text
        return calling.replace('\xa0', ' ').strip()
    
    @exceptor()
    def _get_text(self):
        body = self.soup.find('div', {'class': 'body-block'})
        paragraphs = body.findAll('p')
        text_list = [p.text for p in paragraphs]
        text = '\n\n'.join(text_list)
        
        return text
    
    @exceptor(sep='\n\n----------STARTING NEW TALK SCRAPE----------\n')
    def _main(self):
        self.soup = self._get_soup()
        self.title = self._get_title()
        self.author = self._get_author()
        self.author_calling = self._get_author_calling()
        self.text = self._get_text()
    
    def to_dict(self):
        return dict(
            conference=self.conf,
            title=self.title,
            author=self.author,
            calling=self.author_calling,
            text=self.text,
            lang=self.lang,
            url=self.talk_url 
        )



if __name__ == '__main__':
    LdsConferences = LdsConfScraper(months=['04', '10'], years=range(1971, 2018), delay=1.5)
    conference_df = LdsConferences.to_df()
    output_folder = './output'
    output_filename = 'lds_conferences.csv'
    save_to_file(output_folder, output_filename, conference_df)

    if os.isfile(output_folder + '/' + output_filename):
        #output was written, temp files can be wiped
        pickle_wipe()