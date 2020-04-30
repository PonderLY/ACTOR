import json
import random
import re
import sys
import string
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from pattern.en import lemma


english_stopwords = stopwords.words('english') # The list of English stopwords
# english_stopwords = [str(word) for word in english_stopwords]
english_stopwords = set(english_stopwords)


class POI:
    def __init__(self, poi_id, lat, lng, cat, name):
        self.poi_id = poi_id
        self.lat = lat
        self.lng = lng
        self.cat = cat
        self.name = name

    def __str__(self):
        return '\t'.join([self.name, str(self.lat)+','+str(self.lng), self.cat])

class Tweet:
    def load_tweet(self, line):
        self.line = line
        items = line.split('\x01')
        self.id = long(items[0])
        self.uid = long(items[1])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.datetime = items[4]
        self.ts = int(float(items[5]))%(3600*24)
        self.text = items[6]
        self.words = self.text.split(' ')
        self.raw = items[7]
        self.category = items[9]
        if len(items)>11:
            self.poi_id = items[8]
            self.poi_lat = float(items[9]) if items[9] else items[9]
            self.poi_lng = float(items[10]) if items[10] else items[10]
            self.category = items[11]
            self.poi_name = items[12]


    def load_utgeo(self, line):
        """
        Load dataset UTGEO-2011
        """
        self.line = line
        items = line.split('\t')
        self.name = items[0]
        # self.ts = float(items[1])
        self.ts = int(float(items[1]))%(3600*24)
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.text = items[8]
        self.mention = self.extract_username(self.text)  # A list of mentioned user
        # self.words = self.text.split(' ')
        self.textp = self.punctuate_tweet(self.text)
        self.words = self.lemmatize_tweet(self.textp)
        self.category = ''  # I do not know what it stands for.
    

    def punctuate_tweet(self, text):
        """
        Remove the punctuations of the text.
        Return the list of words splitted by ' '.
        """
        f= lambda x: ''.join([i for i in x if i not in string.punctuation])
        return f(text).split(' ')


    def lemmatize_tweet(self, words):
        """
        Remove the stopwords & username and lemmatize the words.
        Return the list of the lemmas.
        """
        lemmas = []
        words_filtered_stopwords = [word for word in words if not (word in english_stopwords)]
        for word in words_filtered_stopwords:
            lemmas.append(lemma(word))
        return lemmas


    def extract_username(self, text):
        mentioned_list = re.findall(r"@(.+?) ", text)
        for username in mentioned_list:
            if username.startswith(' '):
                mentioned_list.remove(username)
        return mentioned_list

    def load_old_ny(self, line):
        items = line.split('\x01')
        self.id = long(items[0])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.ts = int(float(items[5]))
        self.text = items[6]
        self.words = self.text.split(' ')
        self.category = ''

    def load_checkin(self, line):
        items = line.split('\x01')
        self.id = long(items[0])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.ts = int(float(items[5]))
        self.text = items[7]
        self.words = self.text.split()
        self.category = items[6]

    # load a clean tweet from a mongo database object
    def load_from_mongo(self, d):
        self.id = d['id']
        self.uid = d['uid']
        self.created_at = d['time']
        self.ts = d['timestamp']
        self.lat = d['lat']
        self.lng = d['lng']
        self.text = d['text']
        # self.words = d['words']
        self.words = d['phrases']
