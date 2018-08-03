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

lines = open('D:\Masters\my_project\chatbot\chatbot\movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('D:\Masters\my_project\chatbot\chatbot\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

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
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
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
            
            
# creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        decoding_scope.resuse_variables()
        test_predictions = decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, word2int['<SOS>'],word2int['<EOS>'], sequence_length-1, num_words, decoding_scope, output_function, keep_prob, batch_size)
        
        return training_predictions, test_predictions
        
# Building the seq2seq Model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    # putting together, encoder state and decder return training and test predictions
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words+1, encoder_embedding_size, initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length) #output of encoder neural netwrk
    #get preprocessed targets for the training.
    preprocessed_targets  = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0 ,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, questions_num_words, sequence_length, rnn_size, num_layers, questionswords2int, keep_prob, batch_size) 
    return training_predictions, test_predictions     
            

# setting the Hyperparameters
    
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size= 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001             
keep_probability = 0.5

            
#defining a session a tf session, all tf trainning will run             
tf.reset_default_graph()          
session = tf.InteractiveSession()

# Loading Model Input
inputs, targets, lr, keep_prob = model_inputs()

#setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")
            
# getting the shape of the input tensor
input_shape = tf.shape(inputs)

#getting training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size, sequence_length, len(answerswords2int), len(questionswords2int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, questionswords2int)

# seting up the loss error, the optimiser and gradient clipping - gc is tech that will keep gradient btw min value and max value to avoid any vanishing gradient issue

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizermizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None ]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
            
# padding the sequences with pad tokens
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
            
            
#splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index + batch_size
        questions_in_batch = questions[start_index: start_index+batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########
 
 
 
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)