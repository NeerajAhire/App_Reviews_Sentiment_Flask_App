from flask import Flask, render_template, request
import requests
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO
import base64
import string
import nltk


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
      if request.method == 'POST':
           x_in = request.form.get('x_in')
           aid = 'com.quora.android'
           app_n = 'Quora: the knowledge platform'

           rating = "The Rating for Quora: the knowledge platform is- 4"

           rpd = pd.read_csv('file1.csv', index_col=0)

           slist = {'Positive': 1125, 'Negative': 801, 'Neutral': 74}

           with open('file2.txt') as f:
               rpdslst = f.read().splitlines()

           rlist = [rpd['review_rating'].to_list().count(1), rpd['review_rating'].to_list().count(2), rpd['review_rating'].to_list().count(3),
                    rpd['review_rating'].to_list().count(4), rpd['review_rating'].to_list().count(5)]

           fsw = [i.translate(str.maketrans('', '', string.punctuation)) for i in
                  nltk.corpus.stopwords.words('english')]
           app_n = app_n.lower()
           app_n = app_n.translate(str.maketrans('', '', string.punctuation))
           app_n = ''.join([i for i in app_n if not i.isdigit()])
           app_n = " ".join(app_n.split())

           fsw = fsw + app_n.split()
           fsw.append('app')

           wlist = rpd['review_text'].astype('str').to_list()
           wcnt = {}
           pwcnt = {}
           nwcnt = {}
           tbg = {}
           pbg = {}
           nbg = {}

           k=0
           for i in wlist:
               sen = i.lower()
               sen = sen.translate(str.maketrans('', '', string.punctuation))
               sen = ''.join([i for i in sen if not i.isdigit()])
               sen = " ".join(sen.split())
               bg = [b for b in zip(sen.split(" ")[:-1], sen.split(" ")[1:])]
               for n in bg:
                   if n in tbg:
                       tbg[n] += 1
                   else:
                       tbg[n] = 1
                   if rpdslst[k] == 'Positive':
                       if n in pbg:
                           pbg[n] += 1
                       else:
                           pbg[n] = 1
                   elif rpdslst[k] == 'Negative':
                       if n in nbg:
                           nbg[n] += 1
                       else:
                           nbg[n] = 1

               senlist = sen.split()
               senlist = [w for w in senlist if w not in fsw]
               for j in senlist:
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

           tbg = {k: v for k, v in tbg.items() if (k[0] not in fsw and k[1] not in fsw)}
           pbg = {k: v for k, v in pbg.items() if (k[0] not in fsw and k[1] not in fsw)}
           nbg = {k: v for k, v in nbg.items() if (k[0] not in fsw and k[1] not in fsw)}

           tbgc = sorted([(tbg[w], w) for w in tbg], reverse=True)
           pbgc = sorted([(pbg[w], w) for w in pbg], reverse=True)
           nbgc = sorted([(nbg[w], w) for w in nbg], reverse=True)

           img = BytesIO()

           fig = plt.figure(figsize=(15, 10))
           gs = fig.add_gridspec(4, 8, hspace=0.7, height_ratios=[1.7, 1, 1, 1], top=0.94)
           ax1 = fig.add_subplot(gs[0, 0:3])
           ax1.pie([slist['Positive'], slist['Negative'], slist['Neutral']],
                   labels=['Positive', 'Negative', 'Neutral'], autopct='%.0f%%')
           plt.title("Sentiment Pie Chart", fontweight="bold", pad=9)
           ax2 = fig.add_subplot(gs[0, 3:6])
           ax2.pie(rlist, labels=['1', '2', '3', '4', '5'], autopct='%.0f%%')
           plt.title("Ratings Pie Chart", fontweight="bold", pad=9)
           ax3 = fig.add_subplot(gs[1, 0:3])
           ax3.bar([wcntl[i][1] for i in range(5)], [wcntl[i][0] for i in range(5)], color='blue', width=0.4)
           plt.title("Most Frequent Words", fontweight="bold", pad=2)
           ax4 = fig.add_subplot(gs[2, 0:3])
           ax4.bar([pwcntl[i][1] for i in range(5)], [pwcntl[i][0] for i in range(5)], color='green', width=0.4)
           plt.title("Most Frequent Words in Positive Reviews", fontweight="bold", pad=2)
           ax5 = fig.add_subplot(gs[3, 0:3])
           ax5.bar([nwcntl[i][1] for i in range(5)], [nwcntl[i][0] for i in range(5)], color='red', width=0.4)
           plt.title("Most Frequent Words in Negative Reviews", fontweight="bold", pad=2)
           ax6 = fig.add_subplot(gs[1:4, 3:6])
           ax6.barh([' '.join(tbgc[i][1]) for i in reversed(range(5))], [tbgc[i][0] for i in reversed(range(5))], color='purple', height=0.6)
           plt.setp(ax6.yaxis.get_majorticklabels(), rotation=90, ha="center", rotation_mode="anchor")
           ax6.yaxis.set_ticks_position('none')
           plt.title("Most Frequent Bigrams", fontweight="bold", pad=2)

           ax7 = fig.add_subplot(gs[:, 6:8])
           plt.title("Most Liked Reviews", fontweight="bold", pad=2)
           txt = ax7.text(0.055, 0.7, '\"' + rpd.sort_values(['review_likes'], ascending=False)['review_text'].iloc[0] + '\"', fontsize=10, wrap=True)
           txt._get_wrap_line_width = lambda: 250
           txt = ax7.text(0.055, 0.38, '\"' + rpd.sort_values(['review_likes'], ascending=False)['review_text'].iloc[1] + '\"',fontsize=10, wrap=True)
           txt._get_wrap_line_width = lambda: 250
           txt = ax7.text(0.055, 0.02, '\"'+ rpd.sort_values(['review_likes'], ascending=False)['review_text'].iloc[2] + '\"',fontsize=10, wrap=True)
           txt._get_wrap_line_width = lambda: 250
           ax7.xaxis.set_tick_params(labelbottom=False)
           ax7.yaxis.set_tick_params(labelleft=False)
           ax7.set_xticks([])
           ax7.set_yticks([])

           plt.savefig(img, format='png')
           plt.close()
           img.seek(0)
           plot_url = base64.b64encode(img.getvalue()).decode()
           topr = rpd.sort_values(['review_likes'],ascending=False)['review_text'].iloc[0]

           return render_template('index.html', plot_url=plot_url, rating=rating,
                                  topr = topr)

      return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__' :
    app.run()
