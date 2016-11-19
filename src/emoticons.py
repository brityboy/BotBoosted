""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (anyall.org, brenocon@gmail.com)"
__version__= "april 2009"

#from __future__ import print_function
import re,sys

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea +
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

# this is going to be rewritten to have MORE emoticons inside it
def analyze_tweet(text):
  h= Happy_RE.search(text)
  s= Sad_RE.search(text)
  w= Wink_RE.search(text)
  t= Tongue_RE.search(text)
  a= Other_RE.search(text)
  # if h and s: return "BOTH_HS"
  if h: return "HAPPY"
  elif s: return "SAD"
  elif w: return "WINK"
  elif a: return "OTHER"
  elif t: return "TONGUE"
  else: return text
  # return "NA"

  # more complex & harder, so disabled for now
  # h,w,s,t,a = [bool(x) for x in [h,w,s,t,a]]
  # if sum([h,w,s,t,a])>1: return "MULTIPLE"
  # if sum([h,w,s,t,a])==1:
  #  if h: return "HAPPY"
  #  if s: return "SAD"
  #return "NA"

if __name__=='__main__':
  for line in sys.stdin:
    import sane_re
    sane_re._S(line[:-1]).show_match(Emoticon_RE, numbers=False)
    #print(analyze_tweet(line.strip()), line.strip(), sep="\t")
