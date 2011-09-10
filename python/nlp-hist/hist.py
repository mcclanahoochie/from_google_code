#---------- histogram.py ----------#
# Create occurrence counts of words or characters
# A few utility functions for presenting results
# Avoids requirement of recent Python features

from string import split, maketrans, translate, punctuation, digits
import sys
from types import *
import types


def trigram_histogram(source):
      """Create histogram of normalized words (no punct or digits)"""
      hist = {}
      trans = maketrans('','')
      line = source.readline()
      while line:
            prevword1 = ''
            prevword2 = ''
            for word in split(line):
                  word = translate(word, trans, punctuation+digits)
                  if len(word) > 0:
                        word = word.lower()
                        if prevword1 == '':
                              prevword1 = word
                        elif prevword2 == '':
                              prevword2 = word
                        else:
                              trigram = prevword1+' '+prevword2+' '+word
                              hist[trigram] = hist.get(trigram,0) + 1
                              prevword1 = prevword2
                              prevword2 = word
            line = source.readline()
      return hist

def bigram_histogram(source):
      """Create histogram of normalized words (no punct or digits)"""
      hist = {}
      trans = maketrans('','')
      line = source.readline()          
      while line:
            prevword = ''
            for word in split(line):
                  word = translate(word, trans, punctuation+digits)
                  if len(word) > 0:
                        word = word.lower()
                        if prevword == '':
                              prevword = word
                        else:
                              bigram = prevword+' '+word
                              hist[bigram] = hist.get(bigram,0) + 1
                              prevword = word
            line = source.readline()
      return hist

def word_histogram(source):
      """Create histogram of normalized words (no punct or digits)"""
      hist = {}
      trans = maketrans('','')
      line = source.readline()          
      while line:
            for word in split(line):
                  word = translate(word, trans, punctuation+digits)
                  if len(word) > 0:
                        word = word.lower()
                        hist[word] = hist.get(word,0) + 1
            line = source.readline()
      return hist

def most_common(hist, num=1):
      pairs = []
      for pair in hist.items():
            pairs.append((pair[1],pair[0]))
      pairs.sort()
      pairs.reverse()
      return pairs[:num]

def first_things(hist, num=1):
      pairs = []
      things = hist.keys()
      things.sort()
      for thing in things:
            pairs.append((thing,hist[thing]))
      pairs.sort()
      return pairs[:num]

def generate_sentence(hist, num=1):
      pairs = []
      for pair in hist.items():
            pairs.append((pair[1],pair[0]))
      # pairs.sort()
      # pairs.reverse()
      expanded = []
      for pair in pairs:
            cnt = pair[0]
            wrd = pair[1]
            for i in range(cnt):
                  expanded.append(wrd)
      sentence = []
      import random as r
      for j in range(num):
            # rnd = int(r.random()*(len(expanded)-1))
            rnd = int(r.uniform(0,(len(expanded)-1)))
            print rnd
            sentence.append(expanded[rnd])
      return sentence

def sample_sentence(hist, thresh, num=1):
      pairs = []
      for pair in hist.items():
            pairs.append((pair[1],pair[0]))
      # pairs.sort()
      # pairs.reverse()
      sentence = []
      pmin = 10
      pmax = -10
      import random as r
      print '  target prob',thresh
      while (num > 0):
            sample = int(r.uniform(0,(len(pairs)-1)))
            pair = pairs[sample]
            cnt = pair[0]
            wrd = pair[1]
            prob = float(cnt) / (len(pairs)-1)
#            print 'prob',prob,'cnt',cnt,'wrd',wrd,'tot',len(pairs)-1
            if prob > thresh:
                  sentence.append(wrd)
                  num = num - 1
            if prob < pmin:
                  pmin = prob
            if prob > pmax:
                  pmax = prob
      print '   min prob',pmin
      print '   max prob',pmax
      return sentence


if __name__ == '__main__':

      filename = ''
      if len(sys.argv) > 1:
            filename = open(sys.argv[1])
      else:
            print 'filename?'
            exit(0)


      hist1 = word_histogram(open(sys.argv[1]))
      hist2 = bigram_histogram(open(sys.argv[1]))
      hist3 = trigram_histogram(open(sys.argv[1]))

      numtop = 15

      print "most common words:"
      for pair in most_common(hist1, numtop):
            print '\t', pair[1], '\t', pair[0]
      print ''

      print "most common bigrams:"
      for pair in most_common(hist2, numtop):
            print '\t', pair[1], '\t', pair[0]
      print ''

      print "most common trigrams:"
      for pair in most_common(hist3, numtop):
            print '\t', pair[1], '\t', pair[0]
      print ''

      sentlen = 111

      print "sentence from words:"
      print sample_sentence(hist1,0.0009,sentlen/1)
      print ''

      print "sentence from bigrams:"
      print sample_sentence(hist2,0.00006,sentlen/2)
      print ''

      print "sentence from trigrams:"
      print sample_sentence(hist3,0.00002,sentlen/3)
      print ''



#       import numpy as np
#       import pylab as p
#       fig = p.figure()
#       ax = fig.add_subplot(1,1,1)
#       ax.plot(hist.values())
# #      ind = range(len(hist))
# #      ax.set_xticks(ind)
# #      ax.set_xticklabels(hist.keys())
#       print hist.values()[0:5]
#       print hist.keys()[0:5]
#       p.show()


