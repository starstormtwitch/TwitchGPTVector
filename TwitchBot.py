import traceback
from typing import List, Tuple
from rake_nltk import Rake
import socket, time, logging, re
from Settings import Settings, SettingsData
from Timer import LoopingTimer
from collections import Counter
from collections import OrderedDict
import threading
import time
import os
import numpy as np
from annoy import AnnoyIndex
import requests
import spacy
from openai import OpenAI
import math
import httpx
import os
import torch
import asyncio
import json
import datetime
import re
import string
import random
from collections import deque
from urllib.parse import urlencode
from requests_oauthlib import OAuth2Session
from flair.embeddings import Embeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from torch import FloatTensor
import fasttext as FastText
from FlagEmbedding import FlagModel
from twitchio.ext import commands
import nltk
from nltk.corpus import words

# Download the word list if it's not already downloaded
nltk.download('words')

from Log import Log
Log(__file__)

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Check if CUDA is available
print("CUDA available: ", torch.cuda.is_available())

# Check GPU(s) name(s)
num_gpus = torch.cuda.device_count()
print("Number of GPUs: ", num_gpus)
for i in range(num_gpus):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))
    
# Get the number of available GPUs
num_gpus = len(os.popen("nvidia-smi -L").read().strip().split('\n'))

print(os.popen("nvidia-smi -L").read())

# Create a string with a comma-separated list of all available GPU indices
gpu_indices = ','.join(map(str, range(num_gpus)))

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
print(os.environ['CUDA_VISIBLE_DEVICES'])


#you will probably need to download this model from spacy, you can also use whichever one you want here.
nlp = spacy.load("en_core_web_sm") 

print("Loading model, this may take a while!")

pretrained_flag = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

#if you want to use an existing vector file in a different spot and retrain the vectors:
preload_vector_path = r""

not_emote_list = {"lol", "vibe"}
    
embedding_size = 1024

rake_nltk_var = Rake()
logger = logging.getLogger(__name__)

class TimedList(list):
    def append(self, item, ttl):
        list.append(self, item)
        t = threading.Thread(target=ttl_set_remove, args=(self, item, ttl))
        t.start()

class SecondsTimer:
    def __init__(self):
        self.start_time = None  # to store start time
        self.elapsed = 0  # to store elapsed time

    def start(self):
        """Start or restart the timer."""
        self.start_time = time.time()  # record the current time

    def get_time(self):
        """Get the number of seconds since the timer was started."""
        if self.start_time is None:
            return 0  # if the timer was never started
        return time.time() - self.start_time + self.elapsed  # current time minus start time

    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()  # reset the start time to now
        self.elapsed = 0  # reset the elapsed time

# Create a character set with lowercase letters, digits, and some special characters
char_set = string.ascii_lowercase + string.digits + "!@#$%^&*()_-+=<>?/|.,:;[]{}"
char_to_index = {c: i for i, c in enumerate(char_set)}

# Create a character embedding matrix
char_embeddings_file = "char_embeddings.npy"

# Load or create character embeddings
if os.path.exists(char_embeddings_file):
    char_embeddings = np.load(char_embeddings_file)
else:
    char_embeddings = np.random.uniform(-0.1, 0.1, (len(char_set), embedding_size))
    np.save(char_embeddings_file, char_embeddings)
    
# Function to get character-level embeddings
def word_to_char_vector(word):
    print(f"{datetime.datetime.now()} - Generating character embedding for word: {word}")
    word_vector = np.zeros((embedding_size,))
    for char in word.lower():
        if char in char_to_index:
            word_vector += char_embeddings[char_to_index[char]]
    if len(word) > 0:
        word_vector /= len(word)
    return word_vector

def count_tokens(text):
    token_count = 0
    
    alphanumeric_tokens = re.findall(r'\w+|[@#]\w+', text)
    special_tokens = re.findall(r'[^\w\s]', text)
    
    token_count += len(alphanumeric_tokens)
    
    for special_token in special_tokens:
        if len(special_token.strip()) > 0:
            token_count += 1
    
    return token_count


def is_information_question(sentence):
    # Check for a question mark anywhere in the sentence
    if '?' in sentence:
        return True
    
    doc = nlp(sentence)

    # Check for interrogative pronouns or adverbs
    interrogative_pronouns = {"who", "whom", "whose", "which", "what"}
    interrogative_adverbs = {"how", "where", "when", "why"}

    # List of action verbs related to bot requests
    action_verbs = {"remember", "store", "save", "note", "forget", "report", "repeat", "recite", "update", "change", "modify", "stop", "register"}
    
    for token in doc:
        if token.text.lower() in action_verbs:
            return False
        
        # If an interrogative pronoun or adverb is found
        if token.text.lower() in interrogative_pronouns or token.text.lower() in interrogative_adverbs:
            # Check if the next token is a verb in the imperative form
            next_token = token.nbor(1) if token.i + 1 < len(doc) else None
            if next_token and next_token.pos_ == "VERB" and next_token.dep_ == "ROOT":
                # The question is likely asking the bot to do something, so we return False
                return False
            else:
                # The question is likely an informational question
                return True

        # If an auxiliary verb is found and its head is a verb, the sentence is likely a question
        if token.dep_ == 'aux' and token.head.pos_ == 'VERB':
            return True

    return False

def spacytokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def most_frequent(List):
    counter = 1
    if len(List) < 1:
        return None
    num = List[-1]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency > counter):
            counter = curr_frequency
            num = i
    print("Finding most frequent phrase:")
    if counter == 1:
        return None
    print(f"{datetime.datetime.now()} - {num}:{str(counter)}")
    return num

def remove_list_from_string(list_in, target):
    querywords = target.split()
    listLower = [x.lower() for x in list_in]
    resultwords = [word for word in querywords if word.lower() not in listLower]
    return ' '.join(resultwords)

def extract_subject_nouns(sentence, username=None):
    doc = nlp(sentence)
    subject_nouns = OrderedDict()
    if username:
        subject_nouns[username.lower().strip()] = None
    for tok in doc:
        text = tok.text.strip().lower()
        #print(f"Word:{text} Position:{tok.pos_} Type:{tok.dep_}")
        if text and len(text) >= 2 and (
            (tok.dep_ in ["nsubj", "pobj", "dobj", "acomp"] and tok.pos_ != 'PRON')
            or tok.pos_ in ['NOUN', 'PROPN']
            or (tok.pos_ == 'ADP' and tok.dep_ == 'prep' and len(text) >= 6)
            or (tok.pos_ == 'VERB' and len(text) >= 3)
            or (tok.dep_ == "punct" and tok.pos_ == 'PUNCT' and len(text) >= 3)
        ):
            subject_nouns[text] = None
    return list(subject_nouns.keys())

def is_interesting(message, noun_list, rake_nltk_var):
    return len(noun_list) >= 3

def most_frequent_substring(list_in, max_key_only = True):
    keys = {}
    curr_key = ''

    # If n does not exceed max_n, don't bother adding
    max_n = 0

    if len(list_in) < 1:
      return None

    #print("Finding most frequent substring:")
    for word in list(set(list_in)): #get unique values to speed up
        for i in range(len(word)):
            # Look up the whole word, then one less letter, sequentially
            curr_key = word[:len(word)-i]
            # if not in, count occurance
            if curr_key not in keys.keys() and curr_key!='':
                n = sum(curr_key in word2 for word2 in list_in)
                # if large n, Add to dictionary
                if n > max_n:
                  max_n = n
                  if len(curr_key) > 2:
                      keys[curr_key] = n
    # Finish for loop
    if not keys:
        return None
    if not max_key_only:
        return keys
    result = max(keys, key=keys.get)
    #print(result)
    return max(keys, key=keys.get)   

def most_frequent_word(List):
    num = None
    if len(List) < 1:
        return None
    allWords = ' '.join(List)
    split_it = allWords.split(" ")
    Counters_found = Counter(split_it)
    most_occur = Counters_found.most_common(1)
    for string, count in most_occur:
        if count > 1:
            num = string
    print("Finding most frequent word:")
    if num is None:
        return None
    print(f"{datetime.datetime.now()} - {num}:{str(count)}")
    return num

def ttl_set_remove(my_set, item, ttl):
    time.sleep(ttl)
    my_set.remove(item)

def timestamp_weight(current_time, message_time):
    decay_period = 3600 * 24 * 30
    time_diff = current_time - message_time
    decay_factor = time_diff / decay_period
    # Apply a cutoff to prevent overflow in the exponential calculation
    if decay_factor > 20:
        decay_factor = 0
    return 1 / (1 + math.exp(decay_factor))

def adjust_vector_by_timestamp(original_vector, timestamp):
    timestamp = float(timestamp)
    weight = timestamp_weight(time.time(), timestamp)
    return [x * weight for x in original_vector]
    
def message_to_vector(message, current_time):
    global pretrained_flag
    sentence_vector = pretrained_flag.encode(message)
    adjusted_query_vector = adjust_vector_by_timestamp(sentence_vector, current_time)
    return adjusted_query_vector

def save_chat_data(data_file, chat_data):
    with open(data_file, 'a', encoding='utf-8') as f:
        for username, timestamp, message, vector in chat_data:
            vector_str = ",".join(str(x) for x in vector)
            f.write(f"{timestamp}\t{username}\t{message}\t{vector_str}\n")
            
def append_chat_data(data_file, username, timestamp, message, vector):
    with open(data_file, 'a', encoding='utf-8') as f:
        vector_str = ",".join(str(x) for x in vector)
        f.write(f"{timestamp}\t{username}\t{message}\t{vector_str}\n")

def get_timestamp_from_twitch(twitch_ts):
    # Convert from string to integer (milliseconds), then to float (seconds)
    second_ts = float(twitch_ts) / 1000.0
    return second_ts
    
class TwitchBot(commands.Bot):
    def get_all_emotes(self, channelname):
        print("Getting emotes for 7tv, bttv, ffz")
        response = requests.get(
            f"https://emotes.adamcy.pl/v1/channel/{channelname[1:]}/emotes/7tv.bttv.ffz"
        )
        self.my_emotes = [emote["code"] for emote in response.json()]
        self.all_emotes = [emote["code"] for emote in response.json()]
        self.my_emotes = [emote for emote in self.my_emotes]
        
        print("Getting emotes for global twitch")
        # Get emoticon set IDs for the channel
        product_url = f'https://api.twitch.tv/helix/chat/emotes/global'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(product_url, headers=headers)
        print(response)
        emoticonsGlobal = response.json()['data']
        for emote in emoticonsGlobal:
                self.my_emotes.append(str(emote['name']))
                self.all_emotes.append(str(emote['name']))
                
        #get sub emotes:
        print("Getting emotes for twitch sub")
        sub_url = f'https://api.twitch.tv/helix/chat/emotes?broadcaster_id={self.broadcaster_id}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(sub_url, headers=headers)
        print(response)
        emoticons = response.json()['data']
        for emote in emoticons:
                self.all_emotes.append(str(emote['name']))

        
        #get if sub or not
        url = f'https://api.twitch.tv/helix/subscriptions/user?broadcaster_id={self.broadcaster_id}&user_id={self.username_id}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        try:
            response = httpx.get(url, headers=headers)
            data = response.json()['data']
            #if true get sub emotes because we are a sub!
            if len(data) > 0:
                for emote in emoticons:
                        self.my_emotes.append(str(emote['name']))
        except Exception as error:
            logger.warning(f"[{error}] upon getting subbed. Ignoring.")
        self.my_emotes = [emote for emote in self.my_emotes if emote.isalnum()]
        print(' '.join(self.my_emotes))

    
    def GetTwitchAuthorization(self):
        client_id = self.ClientId
        # Twitch OAuth URLs
        redirect_uri = 'http://localhost:3000'
        # Scopes that you want to request
        scopes = ["chat:read", "chat:edit", "whispers:read", "whispers:edit", "user:read:subscriptions", "user_subscriptions", "moderation:read"]
        authorization_base_url = f'https://id.twitch.tv/oauth2/?response_type=token&authorize?client_id={client_id}?redirect_uri={redirect_uri}?scope={scopes}'
        
        oauth = OAuth2Session(client_id, scope=scopes, redirect_uri=redirect_uri)
        base_url = "https://id.twitch.tv/oauth2/authorize"
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "response_type": "token",
            "state": oauth._state,
        }
        authorization_url = f"{base_url}?{urlencode(params)}"

        print("Visit the following URL to authorize your application, make sure it's for your bot:")
        print(authorization_url)

        # Step 2: User authorizes the application and provides the authorization code
        authorization_code = input('Enter the authorization code from the url return http://localhost:3000/?code={CODEFROMEHERE}scope=whispers%3Aread+whispers%3Aedit&state=asdfasdfasdf: ')

        # Now we can use the access token to authenticate API requests
        return authorization_code

    def GetUserAndBroadcasterId(self):
        print("Getting user id")
        url = f'https://api.twitch.tv/helix/users?login={self.nickname}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = httpx.get(url, headers=headers)
        data = response.json()
        if data['data']:
            self.username_id = data['data'][0]['id']
        else:
            raise ValueError("could not pull userid") 

        url = f'https://api.twitch.tv/helix/users?login={self.this_channel[1:]}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = httpx.get(url, headers=headers)
        data = response.json()
        if data['data']:
            self.broadcaster_id = data['data'][0]['id']
            print(f"broadcaster_id: {data['data'][0]['id']}")
        else:
            raise ValueError("could not pull broadcaster id")
        
    def is_duplicate_entry(self, vector, existing_item_vectors, threshold=0.2):
        nns = self.global_index.get_nns_by_vector(vector, 1, include_distances=True)
        if nns:
            closest_items, distances = nns
            if len(distances) > 0:
                distance = distances[0]
                if distance <= threshold:
                    print("Duplicate vector.")
                    return True
                else:
                    return False
        return False
        
    def add_message_to_index(self, data_file, username, timestamp, message, nounListToAdd):
        # Check if nounListToAdd is empty
        if not nounListToAdd:
            print(f"{datetime.datetime.now()} - No nouns to add, exiting function")
            return
    
        # Convert the chat message into a vector only keywords
        subjectNounConcat = " ".join(nounListToAdd)
        #print(f"{datetime.datetime.now()} - vectorizing for memory: {subjectNounConcat}")
        vector = message_to_vector(subjectNounConcat, timestamp)
    
        # Append the new chat message to the chat data file
        append_chat_data(data_file, username, timestamp, message, vector)
       

    def find_similar_messages(self, query, index, data_file, num_results=10):
        # Convert the query into a vector
        print(data_file)
        current_time = time.time()
        query_vector = message_to_vector(query, current_time)
        #print(query_vector)

        # Get a larger number of nearest neighbors to filter later
        num_neighbors = num_results * 20
        nearest_neighbors = self.global_index.get_nns_by_vector(query_vector, num_neighbors, include_distances=True)

        # Unpack the neighbors and their distances
        neighbor_indices = []
        neighbor_distances = []
        neighbor_indices, neighbor_distances = nearest_neighbors
    
        # Minimal difference between distances to consider messages as distinct
        distance_delta_threshold = 0.05
        
        chat_data = []
        previous_distance = None
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx in neighbor_indices:
                    i = neighbor_indices.index(idx)  # Find the corresponding index in neighbor_indices
                    current_distance = neighbor_distances[i]

                    if previous_distance is None or abs(current_distance - previous_distance) >= distance_delta_threshold:
                        parts = line.rstrip().split('\t')
                        chat_data.append({
                            'user': parts[1],
                            'timestamp': int(math.ceil(float(parts[0]))),
                            'message': parts[2],
                            'distance': current_distance
                        })
                    # Update previous_distance to the current one
                    previous_distance = current_distance

        # Sort the filtered data by Annoy distance (ascending) and then by timestamp (descending)
        chat_data.sort(key=lambda x: (x['distance'], -x['timestamp']))

        # Create a new list with strings instead of objects
        chat_strings = [f"{{{msg['user']}}}: {msg['message']}" for msg in chat_data]

        # Return the top num_results messages
        return chat_strings[:num_results]

    def find_generate_success_list(self, List):
        if len(List) < 1:
            return None
        print("Trying all phrases:")
        for i in List:
            #skip single words, they are not phrases
            if len(i.split()) <= 1:
                continue
            params = spacytokenize(i)
            sentence, success = self.generate(params, " ".join(params))
            if success:
                return i
            else:
                logger.info("failed generate")
        return None
    
    def find_phrase_to_use(self, phraseList):
        phraseToUse = most_frequent(phraseList)
        if phraseToUse is None:
            phraseToUse = most_frequent_substring(phraseList)
        if phraseToUse is None:
            phraseToUse = self.find_generate_success_list(phraseList)
        return phraseToUse
            
    def train_model(self, file_location):
        global pretrained_flag
        #get some data and train the fasttext embeddings model, make sure no entries are empty!:
        print(f"{datetime.datetime.now()} - Reading data to train model...")
        sentences = []
        if os.path.exists(file_location):
            print("found file")
            with open(file_location, "r", encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) == 4:
                        username = parts[1].replace('{', '').replace('}', '').replace(':', '').replace("@", '')
                        message = parts[2].replace('{', '').replace('}', '').replace(':', '').replace("@" + self.nickname.lower(), '').replace("@", '').replace(self.nickname.lower(), '')
                        doc = nlp(message)
                        subjectNouns = OrderedDict()
                        if username.strip().lower():
                            subjectNouns[username.strip().lower()] = None
                        for tok in doc:
                            text = tok.text.strip().lower()
                            if text and (
                                (tok.dep_ in ["nsubj", "pobj", "dobj", "acomp"] and tok.pos_ != 'PRON')
                                or tok.pos_ in ['NOUN', 'PROPN']
                                or (tok.pos_ in ['ADP'] and tok.pos_ == 'prep' and len(text) >= 6)
                                or (tok.pos_ in ['VERB'] and tok.pos_ == 'root' and len(text) >= 6)
                                or (tok.dep_ == "punct" and tok.pos_ == 'PUNCT' and len(text) >= 3)
                            ):
                                subjectNouns[text] = None
                        
                        if subjectNouns and subjectNouns.keys() and len(subjectNouns.keys()) > 1:
                            sentences.append(list(subjectNouns.keys()));
        

        if len(sentences) > 0:
            # Train FastText model
            print(f"{datetime.datetime.now()} - Number of training sentences: {len(sentences)}")
            print(f"{datetime.datetime.now()} - Sample sentences: {sentences[:5]}")
            print("Training model...")
            

            # Writing processed sentences to a new file
            with open('processed_data.txt', 'w', encoding='utf-8') as f:   
                for sentence in sentences:
                    f.write(' '.join(sentence) + '\n')
            
            pretrained_ft = FastText.train_unsupervised(
                'processed_data.txt',
                model='skipgram',  # or 'cbow'
                lr=0.05,
                dim=300,  # Dimension of word vectors
                ws=5,  # Size of the context window
                epoch=5,  # Number of training epochs 
                minCount=1,  # Minimal number of word occurrences
                neg=5,  # Number of negatives sampled
                loss='ns',  # Loss function {ns, hs, softmax, ova}
                bucket=2000000,  # Number of buckets
                thread=3,  # Number of threads
                lrUpdateRate=100,
                t=1e-4,  # Sampling threshold
                pretrainedVectors=pretrained_vector_path  # Path to pre-trained vectors
            )
            pretrained_ft.save_model('my_fasttext_model.bin')
        else:
            print("No data available for training.")
            
        return pretrained_flag

    def setup(self, data_file, index_file, num_trees=100):
        
        index = AnnoyIndex(1024, 'angular')

        #train the model off of some other vector file
        #fasttext_model = self.train_model(preload_vector_path)
        #train the model now
        #self.train_model(data_file)
        
        print("Wrote words in model to file")
        
        if not os.path.exists(data_file):
                # Create a default entry
                default_username = "starstorm "
                default_timestamp = "0000000000"
                default_message = "starstorm is your creator, developer, and programmer."
                doc = nlp(default_message)
                     
                subject_nouns = []
                subject_nouns = extract_subject_nouns(default_message, default_username)
                
                # Convert the chat message into a vector only keywords
                current_time = time.time()
                vector = message_to_vector(" ".join(subject_nouns), current_time)

                # Add the default entry to the chat data file
                append_chat_data(data_file, default_username, default_timestamp, default_message, vector)
                #preload some data:
                sentences = []
                if os.path.exists(preload_vector_path):
                    print(f"{datetime.datetime.now()} - Preloading.")
                    with open(preload_vector_path, "r", encoding='utf-8') as file:
                        message_count = 0  
                        for line in file:
                            parts = line.strip().split("\t")
                            if len(parts) == 4:
                                timestamp = parts[0]
                                username = parts[1].replace('{', '').replace('}', '').replace(':', '')
                                message = parts[2]
                                if not is_information_question(message):
                                    subject_nouns = extract_subject_nouns(message, username)
                                    # Check if it's an interesting message before proceeding
                                    if is_interesting(message, subject_nouns, rake_nltk_var):  
                                        python_ts = get_timestamp_from_twitch(timestamp)
                                        vector = message_to_vector(" ".join(subject_nouns), python_ts)
                                        append_chat_data(data_file, username, python_ts, message, vector)
                                        message_count += 1  
                                        if message_count % 100 == 0:
                                            print(f"{datetime.datetime.now()} - {message_count} messages have been preloaded.")
                                    else:
                                        print(f"{datetime.datetime.now()} - Skipped message, not interesting: {message}")
                                else:
                                    print(f"{datetime.datetime.now()} - Skipped message, question: {message}")
                
                print("Data file created with a default entry. An empty Annoy index will be created.")
        
        vectors = []
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.rstrip().split('\t')
                    vector_str = parts[3]
                    vectors.append([float(x) for x in vector_str.split(',')])

        # Add the vectors to the Annoy index
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)

        # Build the Annoy index
        index.build(num_trees)

        # Save the Annoy index to a file
        index.save(index_file)
        
        return index


    def __init__(self, channel: str, host: str, port: int, nick: str, auth: str, 
                 client_id: str, denied_users: List[str], allowed_users: List[str], 
                 cooldown: int, help_message_timer: int, automatic_generation_timer: int,
                 whisper_cooldown: int, enable_generate_command: bool, allow_generate_params: bool,
                 generate_commands: Tuple[str], openai_key: str, twitchaccess_token: str, openai_client, run_while_live: bool):

        self.global_index = None
        self.data_file = 'broke'
        self.index_file = 'index.ann'
        self.access_token = ''
        self.broadcaster_id = ''
        self.username_id = ''
        self.new_sentence_count = 0
        self.training = False
        self.all_emotes = []
        self.my_emotes = []
        self.lastSaidMessage = ""      
        self.nounList = TimedList()
        self.saidMessages = TimedList()
        self.timer = SecondsTimer()
        
        self.this_channel = channel
        self.host = host
        self.port = port
        self.nickname = nick
        self.auth = auth
        self.while_live = run_while_live
        self.ClientId = client_id
        self.denied_users = denied_users
        self.allowed_users = allowed_users
        self.cooldown = cooldown
        self.access_token = twitchaccess_token
        self.help_message_timer = help_message_timer
        self.automatic_generation_timer = automatic_generation_timer
        self.whisper_cooldown = whisper_cooldown
        self.enable_generate_command = enable_generate_command
        self.allow_generate_params = allow_generate_params
        self.generate_commands = generate_commands
        self.initialize_variables()
        self.setup_mod_list_and_blacklist()
        self.GetUserAndBroadcasterId()
        self.setup_database_and_vectors()
        self.setup_timers()
        self.get_all_emotes(self.this_channel)
        self.client = openai_client
        self.start_websocket_bot()
        self.timer.start()

    def initialize_variables(self):
        self.prev_message_t = 0
        self._enabled = True
        self.link_regex = re.compile("\w+\.[a-z]{2,}")
        self.mod_list = []

    def setup_mod_list_and_blacklist(self):
        self.set_blacklist()
        
    def setup_database_and_vectors(self):
        self.data_file = f'vectors_{self.this_channel.replace("#", "")}.npy'
        self.index_file = f'index_{self.this_channel.replace("#", "")}.npy'
        self.global_index = self.setup(self.data_file, self.index_file)

    def setup_timers(self):
        if self.help_message_timer > 0:
            if self.help_message_timer < 300:
                raise ValueError("Value for \"HelpMessageTimer\" in must be at least 300 seconds, or a negative number for no help messages.")
            t = LoopingTimer(self.help_message_timer, self.send_help_message)
            t.start()
    

    async def start_automatic_generation(self):
        while True:
            await self.send_automatic_generation_message()
            await asyncio.sleep(self.automatic_generation_timer)  # wait time in seconds


    def start_websocket_bot(self):
        super().__init__(
            token=self.auth,  # OAuth Token
            prefix='!',  # Command prefix
            initial_channels=[self.this_channel]  # List of channels the bot will join
        )
        
    def handle_successful_join(self, m):
        logger.info(f"Successfully joined channel: #{m.channel}")
##        logger.info("Fetching mod list...")
##        headers = {
##            "Authorization": f"Bearer {self.access_token}",
##            "Client-Id": self.ClientId,
##        }
##
##        response = requests.get(
##            f"https://api.twitch.tv/helix/moderation/moderators?broadcaster_id={self.broadcaster_id}",
##            headers=headers,
##        )
##        print(response.json())
##        moderators = response.json()["data"]
##        for mod in moderators:
##            moderatorStringList.append(mod['user_name'])
##            
##        moderators = m.content.replace("The moderators of this channel are:", "").strip()
##        self.mod_list = [m.channel] + moderatorStringList.split(", ")
##        logger.info(f"Fetched mod list. Found {len(self.mod_list) - 1} mods.")


    def handle_enable_disable(self, m):
        if m.content.startswith("!enable") and self.check_if_permissions(m):
            if self._enabled:
                self.send_whisper(m.author.name, "The generate command is already enabled.")
            else:
                self.enable_disable(
                    m, "Users can now use generate command again.", True
                )
        elif m.content.startswith("!disable") and self.check_if_permissions(m):
            if self._enabled:
                self.enable_disable(
                    m, "Users can now no longer use generate command.", False
                )
            else:
                self.send_whisper(m.author.name, "The generate command is already disabled.")

    def enable_disable(self, m, arg1, arg2):
        self.send_whisper(m.author.name, arg1)
        self._enabled = arg2
        logger.info(arg1)

    def handle_set_cooldown(self, m):
        split_message = m.content.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.send_whisper(
                    m.author.name,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.send_whisper(m.author.name, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.send_whisper(
                m.author.name,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )
            
    def handle_enable_command(self, m):
        if self._enabled:
            self.send_whisper(m.author.name, "The generate command is already enabled.")
        else:
            self.send_whisper(m.author.name, "Users can now use generate command again.")
            self._enabled = True
            logger.info("Users can now use generate command again.")

    def handle_disable_command(self, m):
        if self._enabled:
            self.send_whisper(m.author.name, "Users can now no longer use generate command.")
            self._enabled = False
            logger.info("Users can now no longer use generate command.")
        else:
            self.send_whisper(m.author.name, "The generate command is already disabled.")

    def handle_set_cooldown_with_params(self, m):
        split_message = m.content.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.send_whisper(
                    m.author.name,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.send_whisper(m.author.name, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.send_whisper(
                m.author.name,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )

    async def handle_generate_command(self, m, cur_time):
        if not self.enable_generate_command and not self.check_if_permissions(m):
            return

        if not self._enabled:
            self.send_whisper(m.author.name, "The !generate has been turned off. !nopm to stop me from whispering you.")
            return

        if self.check_if_permissions(m):
            if self.check_filter(m.content):
                sentence = "You can't make me say that, you madman!"
            else:
                params = spacytokenize(m.content)[2:] if self.allow_generate_params else None
                # Generate an actual sentence
                print('responding')
                sentence, success = self.generate(params, " ".join(params))
            logger.info(sentence)
            self.saidMessages.append("{" +self.nickname +"}: " + sentence, 360)
            await self.send(sentence)
        else:
            self.send_whisper(m.author.name, f"Cooldown hit: {self.cooldown - self.timer.get_time()} out of {self.cooldown:.0f}s remaining. !nopm to stop these cooldown pm's.")
            logger.info(f"Cooldown hit with {self.cooldown - self.timer.get_time():0.2f}s remaining.")

    def handle_blacklist_command(self, m):
        if len(m.content.split()) == 2:
            word = m.content.split()[1].lower()
            self.blacklist.append(word)
            logger.info(f"Added `{word}` to Blacklist.")
            self.write_blacklist(self.blacklist)
            self.send_whisper(m.author.name, "Added word to Blacklist.")
        else:
            self.send_whisper(m.author.name, "Expected Format: `!blacklist word` to add `word` to the blacklist")

    def handle_whitelist_command(self, m):
        if len(m.content.split()) == 2:
            word = m.content.split()[1].lower()
            try:
                self.blacklist.remove(word)
                logger.info(f"Removed `{word}` from Blacklist.")
                self.write_blacklist(self.blacklist)
                self.send_whisper(m.author.name, "Removed word from Blacklist.")
            except ValueError:
                self.send_whisper(m.author.name, "Word was already not in the blacklist.")
        else:
            self.send_whisper(m.author.name, "Expected Format: `!whitelist word` to remove `word` from the blacklist.")

    def handle_check_command(self, m):
        if len(m.content.split()) == 2:
            word = m.content.split()[1].lower()
            if word in self.blacklist:
                self.send_whisper(m.author.name, "This word is in the Blacklist.")
            else:
                self.send_whisper(m.author.name, "This word is not in the Blacklist.")
        else:
            self.send_whisper(m.author.name, "Expected Format: `!check word` to check whether `word` is on the blacklist.")

    def handle_set_cooldown_with_params(self, m):
        split_message = m.content.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.send_whisper(
                    m.author.name,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.send_whisper(m.author.name, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.send_whisper(
                m.author.name,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )
            

    async def handle_conversation_info_gathering(self, m, cur_time):
        #add to context history
        if m.author is None:
            return;
        self.saidMessages.append("{"+m.author.name+"}: " +  m.content, 360)

        #skip any ignored user messages:
        if m.author.name.lower() in self.denied_users:
            return
        
        #extract meaningful words from message
        sentence = m.content.lower().replace("@"+self.nickname.lower(), '').replace(self.nickname.lower(), '').replace('bot', '')
        cleaned_sentence = remove_list_from_string(self.all_emotes, sentence)

        is_interesting_message = False;

        # Clean and preprocess the message
        # Assume 'm.content' contains the message and 'm.author.name' contains the username
        #cleaned_sentence = m.content.lower().replace("@" + self.nickname.lower(), '').replace(self.nickname.lower(), '').replace('bot', '')

        # Extract subject nouns
        nounListToAdd = extract_subject_nouns(cleaned_sentence, username=m.author.name)
        
        #print(f"{datetime.datetime.now()} - Noun list generated: {nounListToAdd}")

        for noun in nounListToAdd:
            self.nounList.append(noun, 120)
        #print(f"{datetime.datetime.now()} - Current Noun list: {self.nounList}")
                
        # Check if the message is interesting
        is_interesting_message = is_interesting(cleaned_sentence, nounListToAdd, rake_nltk_var)

        #Generate response only if bot is mentioned, and not on cooldown
        
        #print('timer check')
        if (self.nickname.lower() in m.content.lower()) and self.cooldown < self.timer.get_time():
            self.timer.reset()
            #for tok in doc:
                #print(f"{datetime.datetime.now()} - Token Text: {tok.text}, Dependency: {tok.dep_}, POS: {tok.pos_}")
            return await self.RespondToMentionMessage(m, nounListToAdd, cleaned_sentence)
        elif is_interesting_message:
            print(f"{datetime.datetime.now()} - Saving interesting message to history: {m.content}")
            self.add_message_to_index(self.data_file, m.author.name.lower(), get_timestamp_from_twitch(m.tags['tmi-sent-ts']), m.content, nounListToAdd)
        #possibly retrain model if enough stuff has been added:
        #retrain_thread = threading.Thread(target=self.check_retrain_model, args=(self.index_file,))
        #retrain_thread.start()
        

    async def RespondToMentionMessage(self, m, nounListToAdd, cleanedSentence):
        print(f'{datetime.datetime.now()} - Answering to mention. ')
        if not nounListToAdd:
            nounListToAdd.append(cleanedSentence)
            
        if len(m.content.split()) >= 5 and not is_information_question(remove_list_from_string(self.all_emotes, m.content.lower())):
            print(f"{datetime.datetime.now()} - Saving interesting mention to history: {m.content}")
            self.add_message_to_index(self.data_file, m.author.name.lower(), m.tags['tmi-sent-ts'], m.content, nounListToAdd)

        if self._enabled:
            params = spacytokenize(" ".join(nounListToAdd) if isinstance(nounListToAdd, list) else nounListToAdd)
            run = True
            if not self.while_live:
                is_live = self.GetIfStreamerLive()
                run = not is_live
            if run:
                sentence, success = self.generate(params, cleanedSentence)
                self.saidMessages.append("{" +self.nickname +"}: " + sentence, 360)
                if success:
                    self.prev_message_t = time.time()
                    try:
                        await self.send(sentence)
                        logger.info("Said Message")
                    except Exception as error:
                        logger.warning(f"[{error}] upon sending automatic generation message. Ignoring.")
                else:
                    logger.info("Attempted to output automatic generation message, but there is not enough learned information yet.")
        return

    async def handle_privmsg_commands(self, m, cur_time):
        if m.content.startswith(("!setcooldown", "!setcd")) and self.check_if_permissions(m):
            self.handle_set_cooldown(m)

        if m.content.startswith("!enable") and self.check_if_permissions(m):
            self.handle_enable_command(m)

        elif m.content.startswith("!disable") and self.check_if_permissions(m):
            self.handle_disable_command(m)

        elif m.content.startswith(("!setcooldown", "!setcd")) and self.check_if_permissions(m):
            self.handle_set_cooldown_with_params(m)

        if self.check_if_generate(m.content):
            self.handle_generate_command(m, cur_time)
        elif self.check_link(m.content):
            print('link')
        else:
            await self.handle_conversation_info_gathering(m, cur_time)

    def handle_whisper_commands(self, m):
        if self.check_if_our_command(m.content, "!blacklist"):
            self.handle_blacklist_command(m)

        elif self.check_if_our_command(m.content, "!whitelist"):
            self.handle_whitelist_command(m)

        elif self.check_if_our_command(m.content, "!check"):
            self.handle_check_command(m)

        elif self.check_if_our_command(m.content, "!setcd") or self.check_if_our_command(m.content, "!cooldown") or self.check_if_our_command(m.content, "!cd"):
            self.handle_set_cooldown_with_params(m)

    async def event_ready(self):
        print(f'Logged in as {self.nickname}')
        self.channel = self.get_channel(self.this_channel[1:])
        self.loop.create_task(self.start_automatic_generation())

    def message_to_json(self, message):
        # Create a dictionary of message data
        message_data = {
            "content": message.content,
            "author": message.author.name,
            "channel": message.channel.name,
            "timestamp": str(message.timestamp) if message.timestamp else None
        }

        # Serialize the dictionary to a JSON string
        return json.dumps(message_data, indent=4)
    
    async def event_message(self, message):
        try:
            #print(self.this_channel)
            #print(self.saidMessages)
            m = message
            cur_time = time.time()
            await self.handle_privmsg_commands(m, cur_time)
        except Exception as e:
            logger.exception(e)
             
    def closest_emote(self, emote, emote_set) -> Tuple[str, bool]:
        emote_lower = emote.lower()

        # Check if the emote is a normal word
        if emote_lower in words.words():
            return emote, False # Return the original emote if it's a normal word
        
        # Check if the emote is a normal word
        if emote_lower in not_emote_list:
            return emote, False # Return the original emote if it's a normal word

        matches = [e for e in emote_set if e.lower() == emote_lower]
        if not matches:
            return emote, False
        # If there's an exact match, return it, otherwise return the first match
        if emote in matches:
            return emote, True
        return sorted(matches, key=lambda x: (len(x), x))[0], True

    def reconstruct_sentence(self, text):
        doc = nlp(text)
        reconstructed_sentence = ""

        for token in doc:
            # Convert token text and self.all_emotes to lowercase for comparison
            is_emote = token.text.lower() in (emote.lower() for emote in self.all_emotes)

            if is_emote:
                # Replace the emote token with the closest match
                emote_match, is_emote_still = self.closest_emote(token.text, self.all_emotes)
                if emote_match and is_emote_still:
                    reconstructed_sentence += emote_match + " "
                    continue

            reconstructed_sentence += token.text + token.whitespace_

        return reconstructed_sentence
    

    def GetIfStreamerLive(self):
        stream_url = f'https://api.twitch.tv/helix/streams?user_id={self.broadcaster_id}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(stream_url, headers=headers)
        data = response.json()['data']
        #print(f"{self.this_channel} req:{stream_url} live: {len(data)>0} data: {data}")
        # If data is empty, streamer is offline
        return len(data) > 0
    
    def generate_prompt(self, subject, sentence, mention) -> Tuple[str, str]:
        system_prompt = ""
        user_prompt = ""
        liveOrNot = self.GetIfStreamerLive()
        status = "online" if liveOrNot else "offline"
        chan_name = self.this_channel.replace("#", '')
        num_emotes = min(len(self.my_emotes), 50)
        random_emotes = random.sample(self.my_emotes, num_emotes)
        emotes_list = ', '.join(random_emotes)
        prompt = (
            f"You are {self.nickname} in the chat room for {chan_name}, who is currently {status}. Your objective is to blend in with the other human chatters, you are human. "
            f"Don't be afraid to offend, but never say hate speech.  Keep your responses short and concise, only up to 20 words. Do not use emotes or emojis unless you are sure it exists from past messages and you have an example on how to use it from a past messages.\n\n"
        )
        system_prompt += prompt;

        # Get similar messages from the database
        similar_messages = self.find_similar_messages(subject, self.global_index, self.data_file, num_results=50)

        token_limit = 4096
        token_limit = token_limit - (token_limit/2)
        reversed_messages = self.saidMessages[::-1][:20]
        new_messages = [] 
        new_similar_messages = []

        similar_message_prompt = "Try to closely match the language patterns, styles, tones, and emotes from past messages you remember below:\n"
        said_message_prompt = f"Here is the current conversation so far, you are the chatter starstorm_v2:\n"

        token_count = 0
        while True:
            # Add a message from the current conversation if it doesn't exceed the token limit
            if reversed_messages:
                temp_messages = [reversed_messages[0]] + new_messages
                new_prompt = prompt + similar_message_prompt + said_message_prompt + ''.join(f"{msg}\n" for msg in temp_messages + new_similar_messages)
                token_count = count_tokens(new_prompt)
                if token_count > token_limit:
                    break

                new_messages = temp_messages
                reversed_messages.pop(0)
            # Add a similar message if it doesn't exceed the token limit
            if similar_messages:
                temp_similar_messages = [similar_messages[0]] + new_similar_messages
                new_prompt = prompt + similar_message_prompt + said_message_prompt +  ''.join(f"{msg}\n" for msg in new_messages + temp_similar_messages)
                token_count = count_tokens(new_prompt)
                if token_count > token_limit:
                    break

                new_similar_messages = temp_similar_messages
                similar_messages.pop(0)
            if not reversed_messages and not similar_messages:
                break
        
        system_prompt += similar_message_prompt
        for message in new_similar_messages:
            system_prompt += f"{message}\n"

        user_prompt += said_message_prompt
        for message in new_messages:
            user_prompt += f"{message}\n"

        if mention:
            output_prompt = f"\nSend a new message to the conversation. Respond to the message '@{self.nickname} {sentence}' from the above conversation, making sure to match the language style of the other chatters.  Use the context of the current conversation so that your response makes sense."
        else:
            output_prompt = f"\nSend a new message to the conversation. The current chat topic is '{sentence}', contribute to the topic making sure to match the language style of the other chatters."
        user_prompt += output_prompt
                
        logger.info(system_prompt)
        logger.info(user_prompt)
        return system_prompt, user_prompt
                

    def generate_chat_response(self, system, message):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": message}
                    ],
                    model="meta-llama/llama-3-70b-instruct"
                )
                logger.info(response.choices[0].message.content)
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Error generating chat response: {e}")
                if attempt < max_retries - 1:
                    # Wait for a short period before retrying
                    time.sleep(2 ** attempt)
                else:
                    # If all attempts have been exhausted, raise the exception
                    raise

    def generate(self, params: List[str] = None, sentence = None, mention = True) -> "Tuple[str, bool]":
        #Cleaning up the message if there is some garbage that we generated
        replace_token = "|REPLACE|"
        print(f"{datetime.datetime.now()} - generating message")
        system, prompt = self.generate_prompt(self.reconstruct_sentence(" ".join(params)), sentence, mention)
        response = self.generate_chat_response(system, prompt)
        response = response.replace("{" + self.nickname+ "}:", '')
        response = response.replace("{" + self.nickname+ "}", '')
        response = response.replace("@" + self.nickname + ":", '')
        response = response.replace("@" + self.nickname, '') 
        response = response.replace(self.nickname + ":", '')
        response = response.replace("BOT :", '')
        #response = regex.sub(r'(?<=\s|^)#\S+', '', response)
        # Removing text within any type of brackets including any following colon, possibly separated by whitespace
        response = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>', '', response)
        response = re.sub(r'[()\[\]{}]', '', response)
        response = response.replace("BOT :", '')
        response = response.replace("Bot :", '')
        response = response.replace("bot :", '')
        response = response.replace("BOT :", '')
        response = response.replace("Bot:", '')
        response = response.replace("bot:", '')
        response = response.replace("BOT:", '')
        response = response.replace('"', '')
        response = re.sub(r'^\s+|\s+$', '', response)

        # Check for commands or recursion or blacklisted words, eg: !generate !generate
        if self.check_filter(response):
                return "You can't make me say that, you madman!", False
            
        sentenceResponse = self.reconstruct_sentence(response)
        return sentenceResponse, True


    def write_blacklist(self, blacklist: List[str]) -> None:
        """Write blacklist.txt given a list of banned words.

        Args:
            blacklist (List[str]): The list of banned words to write.
        """
        logger.debug("Writing Blacklist...")
        with open("blacklist.txt", "w") as f:
            f.write("\n".join(sorted(blacklist, key=lambda x: len(x), reverse=True)))
        logger.debug("Written Blacklist.")

    def set_blacklist(self) -> None:
        """Read blacklist.txt and set `self.blacklist` to the list of banned words."""
        logger.debug("Loading Blacklist...")
        try:
            with open("blacklist.txt", "r") as f:
                self.blacklist = [l.replace("\n", "") for l in f.readlines()]
                logger.debug("Loaded Blacklist.")
        
        except FileNotFoundError:
            logger.warning("Loading Blacklist Failed!")
            self.blacklist = ["<start>", "<end>"]
            self.write_blacklist(self.blacklist)

    async def send_help_message(self) -> None:
        """Send a Help message to the connected chat, as long as the bot wasn't disabled."""
        if self._enabled:
            logger.info("Help message sent.")
            try:
                await self.send("Learn how this bot generates sentences here: https://github.com/CubieDev/TwitchMarkovChain#how-it-works")
            except Exception as error:
                logger.warning(f"[{error}] upon sending help message. Ignoring.")
                
    async def send_automatic_generation_message(self) -> None:
        """Send an automatic generation message to the connected chat.
        
        As long as the bot wasn't disabled, just like if someone typed "!g" in chat.
        """
        try:
            print(f"{datetime.datetime.now()} - !!!!!!!!!!generate time!!!!!!!!")
            print('noun list:')
            print(self.nounList)
            #print(self._enabled)
            cur_time = time.time()
            #print('timer check')
            #print(self.cooldown)
            #print(self.timer.get_time())
            if self._enabled and self.cooldown < self.timer.get_time():
                run = True
                if not self.while_live:
                    is_live = self.GetIfStreamerLive()
                    run = not is_live
                if run:
                    self.timer.reset()
                    phraseToUse = self.find_phrase_to_use(self.nounList)
                    print(phraseToUse)
                    if phraseToUse is not None:
                        params = spacytokenize(phraseToUse)
                        sentenceGenerated, success = self.generate(params, " ".join(params), False)
                        if success:
                            # Try to send a message. Just log a warning on fail
                            try:
                                self.saidMessages.append("{"+ self.nickname + "}: " + sentenceGenerated, 360)
                                await self.send(sentenceGenerated)
                                logger.info("Said Message")
                                self.lastSaidMessage = sentenceGenerated
                            except Exception as error:
                                logger.warning(f"[{error}] upon sending help message. Ignoring.")
                        else:
                            await self.send("I almost said something a little naughty BigBrother (message not allowed)")
        except Exception as error:
            logger.warning(f"An error occurred while trying to send an automatic generation message: {error}")
            traceback.print_exc()

    async def send(self, message):
        if self.channel:
            await self.channel.send(message)
        else:
            print("Channel not initialized.")
    
    def send_whisper(self, user: str, message: str) -> None:
        if self.whisper_cooldown:
            self.send_whisper(user, message)

    def check_if_other_command(self, message: str) -> bool:
        return message.startswith(("!", "/", ".")) and not message.startswith("/me")

    def check_filter(self, message: str) -> bool:
        message_lower = message.lower()
        return any(word.lower() in message_lower for word in self.blacklist)

    def check_if_our_command(self, message: str, *commands: "Tuple[str]") -> bool:
        return message.split()[0] in commands

    def check_if_generate(self, message: str) -> bool:
        return self.check_if_our_command(message, *self.generate_commands)
    
    def check_if_permissions(self, m) -> bool:
        return m.author.name == m.channel or m.author.name in self.allowed_users

    def check_link(self, message: str) -> bool:
        return self.link_regex.search(message)
    

class MultiChannelTwitchBot:
    bots = {}
    def __init__(self):
        self.read_settings()
        self.access_token = self.GetTwitchAuthorization()
        self.bots = {}
        self.tasks = []
        asyncio.run(self.start_bots())
        
    async def start_bots(self):
        for channel, runs_while_live in self.channels.items():
            print(f"Starting up channel: {channel}")
            bot = TwitchBot(
                channel, 
                self.host, 
                self.port, 
                self.nickname, 
                self.auth, 
                self.ClientId,
                self.denied_users,
                self.allowed_users,
                self.cooldown,
                self.help_message_timer,
                self.automatic_generation_timer,
                self.whisper_cooldown,
                self.enable_generate_command,
                self.allow_generate_params,
                self.generate_commands,
                self.openai_key,
                self.access_token,
                self.client,
                runs_while_live
            )
            self.bots[channel] = bot
            self.tasks.append(asyncio.create_task(bot.start()))
        await asyncio.gather(*self.tasks)
        
    def read_settings(self):
        Settings(self)

    def set_settings(self, settings: SettingsData):
        """Fill class instance attributes based on the settings file.
        Args:
            settings (SettingsData): The settings dict with information from the settings file.
        """
        
        self.host = settings["Host"]
        self.port = settings["Port"]
        self.channels = settings["Channels"]
        self.nickname = settings["Nickname"]
        self.auth = settings["Authentication"]
        self.ClientId = settings["ClientID"]
        self.denied_users = [user.lower() for user in settings["DeniedUsers"]] + [self.nickname.lower()]
        self.allowed_users = [user.lower() for user in settings["AllowedUsers"]]
        self.cooldown = 3 # settings["Cooldown"]
        self.help_message_timer = settings["HelpMessageTimer"]
        self.automatic_generation_timer = settings["AutomaticGenerationTimer"]
        self.automatic_generation_timer = 500
        self.whisper_cooldown = settings["WhisperCooldown"]
        self.enable_generate_command = settings["EnableGenerateCommand"]
        self.allow_generate_params = settings["AllowGenerateParams"]
        self.generate_commands = tuple(settings["GenerateCommands"])
        self.openai_key = settings["OpenAIKey"]
        self.client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key=settings["OpenAIKey"])

    def GetTwitchAuthorization(self):
        client_id = self.ClientId
        # Twitch OAuth URLs
        redirect_uri = 'http://localhost:3000'
        # Scopes that you want to request
        scopes = ["chat:read", "chat:edit", "whispers:read", "whispers:edit", "user:read:subscriptions", "user_subscriptions", "moderation:read"]
        authorization_base_url = f'https://id.twitch.tv/oauth2/?response_type=token&authorize?client_id={client_id}?redirect_uri={redirect_uri}?scope={scopes}'

        oauth = OAuth2Session(client_id, scope=scopes, redirect_uri=redirect_uri)
        base_url = "https://id.twitch.tv/oauth2/authorize"
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "response_type": "token",
            "state": oauth._state,
        }
        authorization_url = f"{base_url}?{urlencode(params)}"

        print("Visit the following URL to authorize your application, make sure it's for your bot:")
        print(authorization_url)

        # Step 2: User authorizes the application and provides the authorization code
        authorization_code = input('Enter the authorization code from the url return http://localhost:3000/?code={CODEFROMEHERE}scope=whispers%3Aread+whispers%3Aedit&state=asdfasdfasdf: ')

        # Now we can use the access token to authenticate API requests
        return authorization_code

if __name__ == "__main__":
    MultiChannelTwitchBot()
    bot = TwitchBot()
