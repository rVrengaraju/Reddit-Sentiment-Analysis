#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import sys
import json


__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

_PUNCTUATION = {
    "?": True,
    ".": True,
    ",": True,
    "!": True,
    ":": True,
    ";": True,
}

# You may need to write regular expressions.

def parsedList(text):
    lst = []
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"(?:__|[*#])|\(http:.*?\)|\[|\]|\(https:.*?\)|\[|\]|\(www\..*?\)|\[|\]", "",text)
    text = re.sub(r"https://.*\.[a-z]*|http://.*\.[a-z]*|www\..*\.[a-z]*", "", text)
    listText = text.split()
    for i, val in enumerate(listText):
        if not val.isalpha():
            tempWord = ""
            tempPunc = ""
            lastIndex = len(val) - 1
            left = 0
            right = len(val)-1
            if val[right] in _PUNCTUATION and val[left] in _PUNCTUATION:		#check if puncutation is on either end of token
                fullFlag = False
                while 1:
                    if left > right:
                        fullFlag = True
                        break
                    if val[left] in _PUNCTUATION:
                        tempPunc += val[left]
                        left += 1
                    else:
                        break
                for i, char in enumerate(list(tempPunc)):
                    lst.append(char)
                tempPunc = ""
                if not fullFlag:
                    while val[right] in _PUNCTUATION:
                        tempPunc += val[right]
                        right -= 1  
                    tempPunc = tempPunc[::-1]
                    for i in range(left,right+1):
                        tempWord += val[i]
                    lst.append(tempWord.lower())
                    for i, char in enumerate(list(tempPunc)):
                        lst.append(char)
            elif val[right] in _PUNCTUATION:									#check if punctuation only on right end of token
                puncFlag = True
                while val[right] in _PUNCTUATION:
                    tempPunc += val[right]
                    right -= 1
                tempPunc = tempPunc[::-1]
                for i in range(right+1):
                    tempWord += val[i]
                lst.append(tempWord.lower())
                for i, char in enumerate(list(tempPunc)):
                    lst.append(char)
            elif val[left] in _PUNCTUATION:										#check if punctuation only on left end of token
                while val[left] in _PUNCTUATION:
                    tempPunc += val[left]
                    left += 1
                for i, char in enumerate(list(tempPunc)):
                    lst.append(char)
                for i in range(left, len(val)):
                    tempWord += val[i]
                lst.append(tempWord.lower())
            else:
                lst.append(val.lower())										#the token only has internal punctuation
        else:
            lst.append(val.lower())
    splitList = []
    tempList = []
    firstPunc = True
    i = 0
    while i<len(lst):														#create list of tokens w/o punctuation
        if lst[i] in _PUNCTUATION:
            if firstPunc == True:
                splitList.append(tempList)
                tempList = []
                firstPunc = False
        else:
            if (lst[i] not in _PUNCTUATION):
                tempList.append(lst[i])
                firstPunc = True
            if i == len(lst) - 1:
                splitList.append(tempList)
        i+=1
    lst = " ".join(lst)
    return lst, splitList  


def unigramList(parsedList):
    unigram = []
    parsedList = parsedList.split()
    for val in parsedList:
        if val not in _PUNCTUATION:
            unigram.append(val)
    return " ".join(unigram)

def bigramList(splitList):
    bigram = []
    for i, val in enumerate(splitList):
        if len(val) > 1:
            for j, wordVal in enumerate(val[:-1]):
                bigram.append(wordVal+"_"+val[j+1])
    return (" ".join(bigram))

def trigramList(splitList):
    trigram = []
    for i, val in enumerate(splitList):
        if len(val) > 2:
            for j, wordVal in enumerate(val[:-2]):
                trigram.append(wordVal+"_"+val[j+1]+"_"+val[j+2])
    return(" ".join(trigram))

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:
    finRes = []
    lst, splitList = parsedList(text)
    finRes.append(lst)
    lst = unigramList(lst)
    finRes.append(lst)
    lst = bigramList(splitList)
    finRes.append(lst)
    lst = trigramList(splitList)
    finRes.append(lst)
    return(finRes)

def main():
	if len(sys.argv) == 2:
		filename = open(sys.argv[1], 'r')
		data = filename.readlines()
		for val in data:
			print(sanitize(json.loads(val)['body']))
	else:
		print("wrong args")

if __name__ == "__main__":
	main()
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.
