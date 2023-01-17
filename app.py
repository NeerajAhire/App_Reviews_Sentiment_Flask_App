from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
from google_play_scraper import search, Sort, reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO
import base64
import string
import nltk
nltk.data.path.append('nltk_data')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
      if request.method == 'POST':
           x_in = request.form.get('x_in')
           sr = search(x_in,
                       lang="en",  # defaults to 'en'
                       country="us",  # defaults to 'us'
                       n_hits=1  # defaults to 30 (= Google's maximum)
                       )

           aid = sr[0]['appId']
           app_n = sr[0]['title']
           rating = "The Rating for " + app_n + " is- " + str(sr[0]['score'])

           result, continuation_token = reviews(
               aid,
               lang='en',  # defaults to 'en'
               country='us',  # defaults to 'us'
               sort=Sort.NEWEST,  # defaults to Sort.NEWEST
               count=2000,  # defaults to 100
               filter_score_with=None  # defaults to None(means all score)
           )

           rpd = pd.DataFrame(result)
           slist = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
           rpdslst = []
           for i in rpd['content']:
               sval = SentimentIntensityAnalyzer().polarity_scores(i)
               if sval['compound'] >= 0.05:
                   slist['Positive'] += 1
                   rpdslst.append('Positive')
               elif sval['compound'] <= -0.05:
                   slist['Negative'] += 1
                   rpdslst.append('Negative')
               else:
                   slist['Neutral'] += 1
                   rpdslst.append('Neutral')

           rlist = [rpd['score'].to_list().count(1), rpd['score'].to_list().count(2), rpd['score'].to_list().count(3),
                    rpd['score'].to_list().count(4), rpd['score'].to_list().count(5)]

           fsw = [i.translate(str.maketrans('', '', string.punctuation)) for i in
                  nltk.corpus.stopwords.words('english')]
           app_n = app_n.lower()
           app_n = app_n.translate(str.maketrans('', '', string.punctuation))
           app_n = ''.join([i for i in app_n if not i.isdigit()])
           app_n = " ".join(app_n.split())

           fsw = fsw + app_n.split()
           fsw.append('app')

           wlist = rpd['content'].astype('str').to_list()
           wcnt = {}
           pwcnt = {}
           nwcnt = {}
           k=0
           for i in wlist:
               sen = i.lower()
               sen = sen.translate(str.maketrans('', '', string.punctuation))
               sen = ''.join([i for i in sen if not i.isdigit()])
               sen = " ".join(sen.split())
               senlist = sen.split()
               for j in senlist:
                   if j not in fsw:
                       if j in wcnt:
                           wcnt[j] += 1
                       else:
                           wcnt[j] = 1
                       if rpdslst[k] == 'Positive':
                           if j in pwcnt:
                               pwcnt[j] +=1
                           else:
                               pwcnt[j] = 1
                       if rpdslst[k] == 'Negative':
                           if j in nwcnt:
                               nwcnt[j] +=1
                           else:
                               nwcnt[j] = 1
               k +=1

           wcntl = sorted([(wcnt[w], w) for w in wcnt], reverse=True)
           pwcntl = sorted([(pwcnt[w], w) for w in pwcnt], reverse=True)
           nwcntl = sorted([(nwcnt[w], w) for w in nwcnt], reverse=True)

           img = BytesIO()

           fig = plt.figure(figsize=(5,7))
           gs = fig.add_gridspec(4, 2, hspace=0.7, height_ratios=[1.7, 1, 1, 1],top=0.94)
           ax1 = fig.add_subplot(gs[0, 0])
           ax1.pie([slist['Positive'], slist['Negative'], slist['Neutral']],
                   labels=['Positive', 'Negative', 'Neutral'], autopct='%.0f%%')
           plt.title("Sentiment Pie Chart", fontweight="bold", pad=9)
           ax2 = fig.add_subplot(gs[0, 1])
           ax2.pie(rlist, labels=['1', '2', '3', '4', '5'], autopct='%.0f%%')
           plt.title("Ratings Pie Chart", fontweight="bold", pad=9)
           ax3 = fig.add_subplot(gs[1, :])
           ax3.bar([wcntl[i][1] for i in range(5)], [wcntl[i][0] for i in range(5)], color='blue', width=0.4)
           plt.title("Most Frequent Words", fontweight="bold", pad=2)
           ax4 = fig.add_subplot(gs[2, :])
           ax4.bar([pwcntl[i][1] for i in range(5)], [pwcntl[i][0] for i in range(5)], color='green', width=0.4)
           plt.title("Most Frequent Words in Positive Reviews", fontweight="bold", pad=2)
           ax5 = fig.add_subplot(gs[3, :])
           ax5.bar([nwcntl[i][1] for i in range(5)], [nwcntl[i][0] for i in range(5)], color='red', width=0.4)
           plt.title("Most Frequent Words in Negative Reviews", fontweight="bold", pad=2)

           plt.savefig(img, format='png')
           plt.close()
           img.seek(0)
           plot_url = base64.b64encode(img.getvalue()).decode()

           return render_template('index.html', plot_url=plot_url, rating=rating)

      return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
   app.run()

