# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:07:24 2018

@author: Aditya
Buildig ChatBot usinng deep NLP
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
            
            
###### Building the seq2seq nodel #######

# creating placeholders for the inputs and the targets
# In tensorflow, all variables are usid in tensors. 
# tensors are like an advanced array, more advanceed than a numpy array, which is of a single type
# and allows fastest computations in deep neural network. Tensorflow all the vairable used in tensor
# must be defined as tensorflow palceholder. this is a kind of more advanced data structure that can contain tensors and other featuires
# hence our first step in any dnlp is to create some placeholders for the inputs and the targets

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name='input')  #tensorflow placeholder(type_of_data, dimensions_of matrix_ofinputdata, name)
    targets = tf.placeholder(tf.int32, [None,None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #Controls drpout rate
    return inputs,targets,lr,keep_prob

# Before creating encoding layer of decoding layer we have to preprocess the targets
#we will feed the neural network with batches of 10 answers at a time.
# each of the answers in batch of target must start with SOS token
# s before creating bathces of answers, we need to put sos in start of each answer

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets            
            
# Encoder RNN layer
#basic lstm cell class by tensorflow
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #apply dropout to lstm -- deactivating certain percentage of neuron....
    #keep_prob will conrol the dropout rate
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    # encoder cell consist of several lstm dropout layer
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, sequence_length = sequence_length, inputs = rnn_inputs, dtype = tf.float32) ## build independent frwrd and bckwrd rnn
    return encoder_state
            
#decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #attention keys - keys to be compared with the target states
    #attention values - values that will be use to construct context vectors
    #attention score - use to get similarity between keys and target states
    #attention contruct - use to build attenstino state
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name= "attn_dec_train")
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
            
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)        
            
#decoding test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, max_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #attention keys - keys to be compared with the target states
    #attention values - values that will be use to construct context vectors
    #attention score - use to get similarity between keys and target states
    #attention contruct - use to build attenstino state
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, decoder_embeddings_matrix, sos_id, eos_id, max_length, num_words, name= "attn_dec_inf")
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function, scope = decoding_scope)
    
    return test_predictions          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            