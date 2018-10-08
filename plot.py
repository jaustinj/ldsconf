import datetime
import re

import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import PorterStemmer
import numpy as np

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import pandas as pd
pd.options.display.max_colwidth = 10000

from scipy import signal

import matplotlib.pyplot as plt

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


    def clean(string):
    string = string.replace('\\n', '')
    string = string.replace('\n', '')
    string = re.sub(r'n\d{1,2}', '', string)
    return string
    
def tokenize(string):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(string)

def unstop(low):
    stops = set(stopwords.words('english'))
    stops.add('n')
    return [word for word in low if word not in stops]

def lowercase(low):
    return [word.lower() for word in low]

def stem(low):
    ps = PorterStemmer()
    return [ps.stem(word) for word in low]

def prepare_text(text):
    text = clean(text)
    words = tokenize(text)
    words = unstop(words)
    words = lowercase(words)
    words = stem(words)
    
    return words

def ngram_prob_dict(text, ngram_num):
    words = prepare_text(text)
    ng = ngrams(words, ngram_num)
    freq = nltk.FreqDist(ng)
    return {key: value / sum(freq.values()) for key, value in freq.items()}

def train_ngrams(df, low_n, high_n):
    df = df
    for n in range(low_n, high_n + 1):
        df['ngram_{}'.format(n)] = df.text.apply(lambda x: ngram_prob_dict(x, n))
        
    return df
    

def text_by_col(df, col):
    return df.groupby(df[col])['text'].apply(lambda s: '{}'.format(''.join(str(s)))).reset_index()

def search_dict(x, string):
    if search in x:
        return x[search]
    else:
        return 
    
    
def plot_ngram(string, window_len=12, window='blackman', ):
    words = prepare_text(string)
    search = tuple(words)
    n = len(words)
    col = 'ngram_{}'.format(n)
    
    if n == 0:
        raise('Must ask for at least one non-stopword')
        
    return_df = ngram_df[['conference', col]].copy()
    
    return_df[col] = return_df[col].apply(
        lambda x: x[search] if search in x else 0.0
    )
    
    return_df.conference = return_df.conference.apply(
        lambda x: datetime.datetime.strptime(x, '%B %Y')
    )
    
    return_df = return_df.sort_values(by='conference')
    conference_num = len(return_df.conference)
    
    smoothed_raw = smooth(return_df[col], window=window, window_len=window_len)
    diff = (len(smoothed_raw) - conference_num) // 2
    smoothed = smoothed_raw[diff:len(smoothed_raw)-diff]
    smoothed = smoothed[:conference_num]
    
    
    trace1 = go.Scatter(
      x=return_df.conference,
      y=return_df[col],
      name='Trend'
    )
    
    trace2 = go.Scatter(
      x=return_df.conference,
      y=smoothed,
      name='Smoothed Trend'
    )
        
    layout = go.Layout(
        showlegend=True
    )
        
    
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)

    return iplot(fig)
    

if __name__ == '__main__':
	df = pd.read_csv('../output/lds_conferences.csv')
	docs = text_by_col(df, 'conference')

	ngram_df = train_ngrams(docs, 1, 4)
	ngram_df.to_pickle('save_model.pkl')

	#plot in jupyter
	#plot_ngram('jesus christ', 12)