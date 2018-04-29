# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:07:24 2018

@author: Aditya
Buildig ChatBot usinng eep NLP
"""
#Libraries
import numpy as np
import tensorflow as tf
import re
import time

#importing dataset

lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

#Creating dictionary that maps each line with its id
id2line = {}

for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line)==5:
        id2line[_line[0]]=_line[-1]
        
# creating list of all of the converstions
conversations_ids=[]
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

#getting separately the questions and the answers
questions=[]
answers=[]
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#Doing a first cleaning of the data
#make every thing in lowercase
#remove all the apostrophe like - that's -> that is
def clean_text(text):
    text = text.lower()
    text= re.sub(r"i'm","i am", text)
    text= re.sub(r"he's","he is", text)
    text= re.sub(r"she's","she is", text)
    text= re.sub(r"that's","that is", text)
    text= re.sub(r"what's","what is", text)
    text= re.sub(r"where's","where is", text)
    text= re.sub(r"\'ll"," will", text)
    text= re.sub(r"\'ve"," have", text)
    text= re.sub(r"\'re"," are", text)
    text= re.sub(r"\'d"," would", text)
    text= re.sub(r"won't","will not", text)
    text= re.sub(r"can't","cannot", text)
    text= re.sub(r"[-()\"#@;:<>{}+=~|.?,]","", text)
    return text

# Cleaning Question
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
    
# Cleaning Answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurences. We will remove
# the words appearing less than 5% of the whole corpus.
word2count={}
for question in clean_questions:
    for word in question.split():
        if not word in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for answer in clean_answers:
    for word in answer.split():
        if not word in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
# Creating two dictionaries that map the question words and the answers word with an integer.
# This is impotant for our seq2seqmodel and a common practise in NLP
threshold=20
questionswords2int={}
word_number=0
for word,count in word2count.items():
    if count>=threshold:
        questionswords2int[word]= word_number
        word_number+=1
answerswords2int={}
word_number=0
for word,count in word2count.items():
    if count>=threshold:
        answerswords2int[word]= word_number
        word_number+=1
        
# we will add last tokens to our two created dictionaries. These last tokens are useful for the encoder and the decoder
# in the seq2seq model which will be the startOF strinf(SOS) and endofstring (EOS). we will use "out" to replace all the
# the words which are less than the threshold as we have them in our previous dictionaries.

tokens=['<PAD>', '<EOS>', '<OUT>', '<SOS>'] #list of final tokens. order is important here.
#now we will add these tokens to our dictionaries
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1
for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1
    
#creating inverse dictionary of the answerswords2int dictionary
#we need this because we need an inverse mapping of the answer int to words in impelementation of seq2seq model
answersint2words = {w_i:w for w,w_i in answerswords2int.items()}

# now we need to add EOS to end of the every answers
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
#translating all the questions and answers into intergers and replacing all the words
# that were filtered by '<OUT>'
questions_to_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word in questionswords2int:
            ints.append(questionswords2int[word])
        else:
            ints.append(questionswords2int['<OUT>'])
    questions_to_int.append(ints)
answers_to_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word in answerswords2int:
            ints.append(answerswords2int[word])
        else:
            ints.append(answerswords2int['<OUT>'])
    answers_to_int.append(ints)

#sorting questions and answers by the length of questions. This will speed up the training.
# it will reduce the amount of padding during the training/
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1, 25+1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])