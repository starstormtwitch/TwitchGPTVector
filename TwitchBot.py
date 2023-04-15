import traceback
from typing import List, Tuple
from TwitchWebsocket import Message, TwitchWebsocket
from rake_nltk import Rake
import socket, time, logging, re
from Settings import Settings, SettingsData
from Timer import LoopingTimer
from Tokenizer import tokenize
from collections import Counter
import threading
import time
import os
import numpy as np
from annoy import AnnoyIndex
import requests
import spacy
import openai
import math
import httpx
import random
from urllib.parse import urlencode
from requests_oauthlib import OAuth2Session
from flair.embeddings import Embeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from torch import FloatTensor

from Log import Log
Log(__file__)

question_words = ["what", "why", "when", "where", "how", "do", "does", 
             "which", "could", "would", "whom", "whose", "?"]

#you will probably need to download this model from spacy, you can also use whichever one you want here.
nlp = spacy.load("en_core_web_sm")

def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
            except ValueError as e:
                print(f"Error details for embedding model line: {e}")
    print(f"{len(glove_model)} words loaded!")
    return glove_model

# Load pre-trained word embeddings, embedding size should match dimensions of model here. You can use a different one if you like.
glove_model = load_glove_model('glove.840B.300d/glove.840B.300d.txt')
embedding_size = 300

rake_nltk_var = Rake()
logger = logging.getLogger(__name__)

class TimedList(list):
    def append(self, item, ttl):
        list.append(self, item)
        t = threading.Thread(target=ttl_set_remove, args=(self, item, ttl))
        t.start()
  
all_emotes = []
lastSaidMessage = ""      
nounList = TimedList()
saidMessages = TimedList()

def is_question(sentence):
    return any((x in sentence for x in question_words))
    
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
    print(f"{num}:{str(counter)}")
    return num

def remove_list_from_string(list_in, target):
    querywords = target.split()
    listLower = [x.lower() for x in list_in]
    resultwords = [word for word in querywords if word.lower() not in listLower]
    return ' '.join(resultwords)
    

def most_frequent_substring(list_in, max_key_only = True):
    keys = {}
    curr_key = ''

    # If n does not exceed max_n, don't bother adding
    max_n = 0

    if len(list_in) < 1:
      return None

    print("Finding most frequent substring:")
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
    print(result)
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
    print(f"{num}:{str(count)}")
    return num

def ttl_set_remove(my_set, item, ttl):
    time.sleep(ttl)
    my_set.remove(item)
        
def message_to_vector(message):
    words = message.split()
    sentence_vector = np.zeros(embedding_size)
    num_words = 0
    for word in words:
        if word in glove_model:
            sentence_vector += glove_model[word]
            num_words += 1
    if num_words > 0:
        sentence_vector /= num_words
    return sentence_vector

def save_chat_data(data_file, chat_data):
    with open(data_file, 'a', encoding='utf-8') as f:
        for username, timestamp, message, vector in chat_data:
            vector_str = ",".join(str(x) for x in vector)
            f.write(f"{timestamp}\t{username}\t{message}\t{vector_str}\n")
            
def append_chat_data(data_file, username, timestamp, message, vector):
    with open(data_file, 'a', encoding='utf-8') as f:
        vector_str = ",".join(str(x) for x in vector)
        f.write(f"{timestamp}\t{username}\t{message}\t{vector_str}\n")

class TwitchBot:
    global_index = None
    data_file = 'broke'
    index_file = 'index.ann'
    access_token = ''
    broadcaster_id = ''
    user_id = ''
    
    def get_all_emotes(self, channelname):
        global all_emotes
        print("Getting emotes for 7tv, bttv, ffz")
        response = requests.get(
            f"https://emotes.adamcy.pl/v1/channel/{channelname[1:]}/emotes/7tv.bttv.ffz"
        )
        all_emotes = [emote["code"] for emote in response.json()]
        all_emotes = [emote for emote in all_emotes if len(emote) >= 3]
        
        print("Getting emotes for global twitch")
        # Get emoticon set IDs for the channel
        product_url = f'https://api.twitch.tv/helix/chat/emotes/global'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(product_url, headers=headers)
        print(response)
        emoticons = response.json()['data']
        for emote in emoticons:
                all_emotes.append(str(emote['name']))
        
        #get if sub or not
        url = f'https://api.twitch.tv/helix/subscriptions/user?broadcaster_id={self.broadcaster_id}&user_id={self.user_id}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        try:
            response = httpx.get(url, headers=headers)
            data = response.json()['data']
            #if true get sub emotes because we are a sub!
            if len(data) > 0:
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
                        all_emotes.append(str(emote['name']))
        except Exception as error:
            logger.warning(f"[{error}] upon getting subbed. Ignoring.")
        print(' '.join(all_emotes))

    
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
        user_id = ''
        url = f'https://api.twitch.tv/helix/users?login={self.nick}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = httpx.get(url, headers=headers)
        data = response.json()
        if data['data']:
            self.user_id = data['data'][0]['id']
        else:
            raise ValueError("could not pull userid") 

        url = f'https://api.twitch.tv/helix/users?login={self.chan[1:]}'
        headers = {
            'Client-ID': self.ClientId,
            'Authorization': f'Bearer {self.access_token}'
        }
        response = httpx.get(url, headers=headers)
        data = response.json()
        if data['data']:
            self.broadcaster_id = data['data'][0]['id']
        else:
            raise ValueError("could not pull broadcaster id") 
        
    def add_message_to_index(self, data_file, username, timestamp, message):
        doc = nlp(message)
        subjectNouns = [tok.text for tok in doc if ((tok.dep_ == "nsubj" or tok.dep_ == "pobj" or tok.dep_ == "acomp" or tok.pos_ == "NOUN") and tok.pos_ != 'PRON')]
        subjectNouns.append(username)
        # Convert the chat message into a vector only keywords
        vector = message_to_vector(" ".join(subjectNouns))

        # Append the new chat message to the chat data file
        append_chat_data(data_file, username, timestamp, message, vector)

        existing_item_vectors = [
            self.global_index.get_item_vector(i)
            for i in range(self.global_index.get_n_items())
        ]
        # Add the new vector to the list of existing item vectors
        existing_item_vectors.append(vector)

        new_index = AnnoyIndex(embedding_size, 'angular')
        for i, v in enumerate(existing_item_vectors):
            new_index.add_item(i, v)

        # Close the existing index file before saving the new index
        self.global_index.unload()

        new_index.build(50)
        # Save the Annoy index to the index file
        if os.path.exists(self.index_file) and os.access(self.index_file, os.R_OK):
            # The file exists and is readable, so we can use it
            new_index.save(self.index_file)
        else:
            # The file does not exist or is not readable, so we need to handle the error
            print("Error: Unable to access index file")

        # Load the new index into the global index
        self.global_index.load(self.index_file)
                
    def find_similar_messages(self, query, index, data_file, num_results=50):
        # Convert the query into a vector
        print(query)
        query_vector = message_to_vector(query)

        # Get a larger number of nearest neighbors to filter later
        num_neighbors = num_results * 5
        nearest_neighbors = self.global_index.get_nns_by_vector(query_vector, num_neighbors, include_distances=True)

        # Unpack the neighbors and their distances
        neighbor_indices = []
        neighbor_distances = []
        neighbor_indices, neighbor_distances = nearest_neighbors

        # Read the chat data
        chat_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, dist in zip(neighbor_indices, neighbor_distances):
                parts = lines[i].rstrip().split('\t')
                chat_data.append({
                    'user': parts[1],
                    'timestamp': int(math.ceil(float(parts[0]))),
                    'message': parts[2],
                    'distance': dist
                })

        # Sort the filtered data by Annoy distance (ascending) and then by timestamp (descending)
        chat_data.sort(key=lambda x: (x['distance'], -x['timestamp']))

        # Return the top num_results messages
        return chat_data[:num_results]

    def find_generate_success_list(self, List):
        if len(List) < 1:
            return None
        print("Trying all phrases:")
        for i in List:
            #skip single words, they are not phrases
            if len(i.split()) <= 1:
                continue
            params = tokenize(i)
            sentence, success = self.generate(params)
            if success:
                return i
            else:
                logger.info("Attempted to output automatic generation message, but there is not enough learned information yet.")
        return None
    
    def find_phrase_to_use(self, phraseList):
        phraseToUse = most_frequent(phraseList)
        if phraseToUse is None:
            phraseToUse = most_frequent_substring(phraseList)
        if phraseToUse is None:
            phraseToUse = self.find_generate_success_list(phraseList)
        return phraseToUse

    def setup(self, vector_dim, data_file, index_file, num_trees=10):
        index = AnnoyIndex(vector_dim, 'angular')
        if not os.path.exists(data_file):
                # Create a default entry
                default_username = "starstorm "
                default_timestamp = "0000000000"
                default_message = "starstorm is your creator, developer, and programmer."
                default_vector = [0.0] * vector_dim

                # Add the default entry to the chat data file
                append_chat_data(data_file, default_username, default_timestamp, default_message, default_vector)

                print("Data file created with a default entry. An empty Annoy index will be created.")
                
        if not os.path.exists(index_file):
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
        # Load the Annoy index from the saved file
        index.load(index_file)

        return index


    def __init__(self):
        self.initialize_variables()
        self.setup_mod_list_and_blacklist()
        self.read_settings()
        self.access_token = self.GetTwitchAuthorization()
        self.GetUserAndBroadcasterId()
        self.setup_database_and_vectors()
        self.setup_timers()
        self.get_all_emotes(self.chan)
        self.start_websocket_bot()


    def initialize_variables(self):
        self.prev_message_t = 0
        self._enabled = True
        self.link_regex = re.compile("\w+\.[a-z]{2,}")
        self.mod_list = []

    def setup_mod_list_and_blacklist(self):
        self.set_blacklist()

    def read_settings(self):
        Settings(self)
    
    def set_settings(self, settings: SettingsData):
        """Fill class instance attributes based on the settings file.
        Args:
            settings (SettingsData): The settings dict with information from the settings file.
        """
        self.host = settings["Host"]
        self.port = settings["Port"]
        self.chan = settings["Channel"]
        self.nick = settings["Nickname"]
        self.auth = settings["Authentication"]
        self.ClientId = settings["ClientID"]
        self.denied_users = [user.lower() for user in settings["DeniedUsers"]] + [self.nick.lower()]
        self.allowed_users = [user.lower() for user in settings["AllowedUsers"]]
        self.cooldown = settings["Cooldown"]
        self.help_message_timer = settings["HelpMessageTimer"]
        self.automatic_generation_timer = settings["AutomaticGenerationTimer"]
        self.whisper_cooldown = settings["WhisperCooldown"]
        self.enable_generate_command = settings["EnableGenerateCommand"]
        self.allow_generate_params = settings["AllowGenerateParams"]
        self.generate_commands = tuple(settings["GenerateCommands"])
        openai.api_key = settings["OpenAIKey"]

    def setup_database_and_vectors(self):
        self.data_file = f'vectors_{self.chan.replace("#", "")}.npy'
        self.index_file = f'index_{self.chan.replace("#", "")}.npy'
        self.global_index = self.setup(embedding_size, self.data_file, self.index_file)

    def setup_timers(self):
        if self.help_message_timer > 0:
            if self.help_message_timer < 300:
                raise ValueError("Value for \"HelpMessageTimer\" in must be at least 300 seconds, or a negative number for no help messages.")
            t = LoopingTimer(self.help_message_timer, self.send_help_message)
            t.start()

        if self.automatic_generation_timer > 0:
            if self.automatic_generation_timer < 15:
                raise ValueError("Value for \"AutomaticGenerationMessage\" in must be at least 15 seconds, or a negative number for no automatic generations.")
            t = LoopingTimer(self.automatic_generation_timer, self.send_automatic_generation_message)
            t.start()

    def start_websocket_bot(self):
        self.ws = TwitchWebsocket(host=self.host,
                                  port=self.port,
                                  chan=self.chan,
                                  nick=self.nick,
                                  auth=self.auth,
                                  callback=self.message_handler,
                                  capability=["commands", "tags"],
                                  live=True)
        self.ws.start_bot()

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
##        moderators = m.message.replace("The moderators of this channel are:", "").strip()
##        self.mod_list = [m.channel] + moderatorStringList.split(", ")
##        logger.info(f"Fetched mod list. Found {len(self.mod_list) - 1} mods.")


    def handle_enable_disable(self, m):
        if m.message.startswith("!enable") and self.check_if_permissions(m):
            if self._enabled:
                self.ws.send_whisper(m.user, "The generate command is already enabled.")
            else:
                self.enable_disable(
                    m, "Users can now use generate command again.", True
                )
        elif m.message.startswith("!disable") and self.check_if_permissions(m):
            if self._enabled:
                self.enable_disable(
                    m, "Users can now no longer use generate command.", False
                )
            else:
                self.ws.send_whisper(m.user, "The generate command is already disabled.")

    def enable_disable(self, m, arg1, arg2):
        self.ws.send_whisper(m.user, arg1)
        self._enabled = arg2
        logger.info(arg1)

    def handle_set_cooldown(self, m):
        split_message = m.message.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.ws.send_whisper(
                    m.user,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.ws.send_whisper(m.user, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.ws.send_whisper(
                m.user,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )
            
    def handle_enable_command(self, m):
        if self._enabled:
            self.ws.send_whisper(m.user, "The generate command is already enabled.")
        else:
            self.ws.send_whisper(m.user, "Users can now use generate command again.")
            self._enabled = True
            logger.info("Users can now use generate command again.")

    def handle_disable_command(self, m):
        if self._enabled:
            self.ws.send_whisper(m.user, "Users can now no longer use generate command.")
            self._enabled = False
            logger.info("Users can now no longer use generate command.")
        else:
            self.ws.send_whisper(m.user, "The generate command is already disabled.")

    def handle_set_cooldown_with_params(self, m):
        split_message = m.message.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.ws.send_whisper(
                    m.user,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.ws.send_whisper(m.user, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.ws.send_whisper(
                m.user,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )

    def handle_generate_command(self, m, cur_time):
        if not self.enable_generate_command and not self.check_if_permissions(m):
            return

        if not self._enabled:
            self.send_whisper(m.user, "The !generate has been turned off. !nopm to stop me from whispering you.")
            return

        if self.prev_message_t + self.cooldown < cur_time or self.check_if_permissions(m):
            if self.check_filter(m.message):
                sentence = "You can't make me say that, you madman!"
            else:
                params = tokenize(m.message)[2:] if self.allow_generate_params else None
                # Generate an actual sentence
                print('responding')
                sentence, success = self.generate(params)
                if success:
                    # Reset cooldown if a message was actually generated
                    self.prev_message_t = time.time()
            logger.info(sentence)
            saidMessages.append("{" +self.nick +"}: " + sentence, 360)
            self.ws.send_message(sentence)
        else:
            self.send_whisper(m.user, f"Cooldown hit: {self.prev_message_t + self.cooldown - cur_time:0.2f} out of {self.cooldown:.0f}s remaining. !nopm to stop these cooldown pm's.")
            logger.info(f"Cooldown hit with {self.prev_message_t + self.cooldown - cur_time:0.2f}s remaining.")

    def handle_blacklist_command(self, m):
        if len(m.message.split()) == 2:
            word = m.message.split()[1].lower()
            self.blacklist.append(word)
            logger.info(f"Added `{word}` to Blacklist.")
            self.write_blacklist(self.blacklist)
            self.ws.send_whisper(m.user, "Added word to Blacklist.")
        else:
            self.ws.send_whisper(m.user, "Expected Format: `!blacklist word` to add `word` to the blacklist")

    def handle_whitelist_command(self, m):
        if len(m.message.split()) == 2:
            word = m.message.split()[1].lower()
            try:
                self.blacklist.remove(word)
                logger.info(f"Removed `{word}` from Blacklist.")
                self.write_blacklist(self.blacklist)
                self.ws.send_whisper(m.user, "Removed word from Blacklist.")
            except ValueError:
                self.ws.send_whisper(m.user, "Word was already not in the blacklist.")
        else:
            self.ws.send_whisper(m.user, "Expected Format: `!whitelist word` to remove `word` from the blacklist.")

    def handle_check_command(self, m):
        if len(m.message.split()) == 2:
            word = m.message.split()[1].lower()
            if word in self.blacklist:
                self.ws.send_whisper(m.user, "This word is in the Blacklist.")
            else:
                self.ws.send_whisper(m.user, "This word is not in the Blacklist.")
        else:
            self.ws.send_whisper(m.user, "Expected Format: `!check word` to check whether `word` is on the blacklist.")

    def handle_set_cooldown_with_params(self, m):
        split_message = m.message.split(" ")
        if len(split_message) == 2:
            try:
                cooldown = int(split_message[1])
            except ValueError:
                self.ws.send_whisper(
                    m.user,
                    "The parameter must be an integer amount, eg: !setcd 30",
                )
                return
            self.cooldown = cooldown
            Settings.update_cooldown(cooldown)
            self.ws.send_whisper(m.user, f"The !generate cooldown has been set to {cooldown} seconds.")
        else:
            self.ws.send_whisper(
                m.user,
                "Please add exactly 1 integer parameter, eg: !setcd 30.",
            )

    def handle_conversation_info_gathering(self, m, cur_time):
        #add to context history
        saidMessages.append("{"+m.user+"}: " +  m.message.lower(), 360)
        
        #extract meaningful words from message
        sentence = m.message.lower().replace(self.nick.lower(), '').replace('bot', '')
        cleanedSentence = remove_list_from_string(all_emotes, sentence)
        doc = nlp(cleanedSentence)
        subjectNouns = [
            tok.text
            for tok in doc
            if tok.dep_ in ["nsubj", "pobj", "acomp"]
            and tok.pos_ != 'PRON'
            or tok.pos_ in ['NOUN', 'PROPN']
        ]
        nounListToAdd = list(set(subjectNouns))
        rake_nltk_var.extract_keywords_from_text(sentence)
        keywordListToAdd = [
            ' '.join(dict.fromkeys(keyword.split(" ")))
            for keyword in rake_nltk_var.get_ranked_phrases()
        ]
        possible_response = []
        for noun in nounListToAdd:
            for keyword in keywordListToAdd:
                if noun in keyword:
                    print(f'interesting noun:{noun}')
                    nounList.append(noun, 15)
                    interestingPhrase = sentence[sentence.lower().find(keyword):][:len(keyword)]
                    possible_response.append(interestingPhrase)
                    break
            else:
                possible_response.append(noun)

        #Generate response only if bot is mentioned, and not on cooldown
        if (self.nick.lower() in m.message.lower() or "bot" in m.message.lower()) and self.prev_message_t + self.cooldown < cur_time:
            return self.RespondToMentionMessage(m, nounListToAdd, cleanedSentence)

    def RespondToMentionMessage(self, m, nounListToAdd, cleanedSentence):
        print('Answering to mention. ')
        if not is_question(remove_list_from_string(all_emotes, m.message.lower())):
            self.add_message_to_index(self.data_file, m.user.lower(), m.tags['tmi-sent-ts'], m.message)

        if not nounListToAdd:
            nounListToAdd.append(cleanedSentence)

        nounListToAdd.append(m.user.lower())

        if self._enabled:
            params = tokenize(" ".join(nounListToAdd) if isinstance(nounListToAdd, list) else nounListToAdd)
            sentence, success = self.generate(params)

            if success:
                self.prev_message_t = time.time()
                try:
                    self.ws.send_message(sentence)
                except Exception as error:
                    logger.warning(f"[{error}] upon sending automatic generation message. Ignoring.")
            else:
                logger.info("Attempted to output automatic generation message, but there is not enough learned information yet.")
        return

    def handle_privmsg_commands(self, m, cur_time):
        if m.message.startswith(("!setcooldown", "!setcd")) and self.check_if_permissions(m):
            self.handle_set_cooldown(m)

        if m.message.startswith("!enable") and self.check_if_permissions(m):
            self.handle_enable_command(m)

        elif m.message.startswith("!disable") and self.check_if_permissions(m):
            self.handle_disable_command(m)

        elif m.message.startswith(("!setcooldown", "!setcd")) and self.check_if_permissions(m):
            self.handle_set_cooldown_with_params(m)

        if self.check_if_generate(m.message):
            self.handle_generate_command(m, cur_time)
        elif self.check_if_other_command(m.message):
            print('command')
        elif self.check_link(m.message):
            print('link')
        else:
            self.handle_conversation_info_gathering(m, cur_time)

    def handle_whisper_commands(self, m):
        if self.check_if_our_command(m.message, "!blacklist"):
            self.handle_blacklist_command(m)

        elif self.check_if_our_command(m.message, "!whitelist"):
            self.handle_whitelist_command(m)

        elif self.check_if_our_command(m.message, "!check"):
            self.handle_check_command(m)

        elif self.check_if_our_command(m.message, "!setcd") or self.check_if_our_command(m.message, "!cooldown") or self.check_if_our_command(m.message, "!cd"):
            self.handle_set_cooldown_with_params(m)

    
    def message_handler(self, m: Message):
        global nounList
        global lastSaidMessage
        global all_emotes
        try:
            if m.type == "366":
                self.handle_successful_join(m)
                
            elif m.type in ("PRIVMSG", "WHISPER"):
                self.handle_enable_disable(m)

                if m.type == "PRIVMSG":
                    cur_time = time.time()
                    self.handle_privmsg_commands(m, cur_time)

            elif m.type == "WHISPER":
                if m.user.lower() in self.mod_list + self.allowed_users:
                    self.handle_whisper_commands(m)

        except Exception as e:
            logger.exception(e)
             
    def reconstruct_sentence(self, text):
        doc = nlp(text)
        tokens = list(doc)
        reconstructed_sentence = ""

        for i, token in enumerate(tokens):
            if token.is_space:
                continue

            is_replace_token = token.text in ('|', 'REPLACE') or (i > 0 and tokens[i - 1].text == '|' and token.text == 'REPLACE')
            is_emote = token.text in all_emotes
            is_prev_emote = i > 0 and tokens[i - 1].text in all_emotes

            if is_emote or is_prev_emote:
                reconstructed_sentence += " "
            elif i > 0 and tokens[i - 1].text[-1] in ["'", "-"] or "'" in token.text or "-" in token.text:
                pass
            elif token.text == "#" or (i > 0 and tokens[i - 1].text == "#"):
                reconstructed_sentence += " "
            elif not token.is_punct and not token.is_left_punct and i > 0 and not is_replace_token:
                reconstructed_sentence += " "

            reconstructed_sentence += token.text

        return reconstructed_sentence

    def generate_prompt(self, subject):
        chan_name = self.chan.replace("#", '')
        num_emotes = min(len(all_emotes), 50)
        random_emotes = random.sample(all_emotes, num_emotes)
        emotes_list = ', '.join(random_emotes)
        prompt = (
            f"Imagine you're a hilarious twitch chatter named bot, chatbot, robot, and {self.nick} lighting up the chat room for {chan_name}. "
            f"Your username is {{{self.nick}}}, only respond as yourself. Use '@username ' to only reply to someone specific. "
            f"Use only these emotes exactly, including their letter case: {emotes_list} . Don't use any punctuation around the emotes. You are not allowed to use any hashtags. " 
            f"The current subject you must respond about is '{subject}'. " 
            f"Unleash your wit in a concise message less than 100 characters: \n"
        )

        # Get similar messages from the database
        similar_messages = self.find_similar_messages(subject, self.global_index, self.data_file, num_results=50)

        token_limit = 4096
        reversed_messages = saidMessages[::-1]
        new_messages = []
        new_similar_messages = []

        similar_message_prompt = "\nYou remember these old messages from the past, DO NOT REPLY to users from this list, and make sure that these inform your response:\n"
        said_message_prompt = "\nThis is the current conversation, ordered from old to new messages, try to reply to a bottom message from this list:\n"

        new_prompt = prompt
        new_prompt += similar_message_prompt
        new_prompt += said_message_prompt
        while True:
            # Add a message from the current conversation if it doesn't exceed the token limit
            if reversed_messages:
                temp_messages = [reversed_messages[0]] + new_messages
                new_prompt = prompt + ''.join(f"{msg}\n" for msg in temp_messages + new_similar_messages)
                token_count = len(spacytokenize(new_prompt))
                if token_count > token_limit:
                    break

                new_messages = temp_messages
                reversed_messages.pop(0)
            # Add a similar message if it doesn't exceed the token limit
            if similar_messages:
                temp_similar_messages = [similar_messages[0]] + new_similar_messages
                new_prompt = prompt + ''.join(f"{msg}\n" for msg in new_messages + temp_similar_messages)
                token_count = len(spacytokenize(new_prompt))
                if token_count > token_limit:
                    break

                new_similar_messages = temp_similar_messages
                similar_messages.pop(0)
            if not reversed_messages and not similar_messages:
                break
        prompt += similar_message_prompt
        # Add the messages to the prompt
        for message in new_similar_messages:
            prompt += f"{{message['user']}}: {message['message']}\n"

        prompt += said_message_prompt
        for message in new_messages:
            prompt += f"{message}\n"

        logger.info(prompt)
        return prompt
                

    def generate_chat_response(self, message):
        
        response = openai.ChatCompletion.create(
            messages= [
                {"role": "user", "content" : message}
                       ],
            model= "gpt-3.5-turbo"
        )
        logger.info(response)
        return response["choices"][0]["message"]["content"]

    def generate(self, params: List[str] = None) -> "Tuple[str, bool]":
        #Cleaning up the message if there is some garbage that we generated
        replace_token = "|REPLACE|"
        prompt = self.generate_prompt(self.reconstruct_sentence(" ".join(params)))
        response = self.generate_chat_response(prompt)
        response = response.replace("@" + self.nick + ":", '')
        response = response.replace(self.nick + ":", '')
        response = response.replace("@" + self.nick, '')
        response = response.replace("BOT :", '')
        #response = regex.sub(r'(?<=\s|^)#\S+', '', response)
        response = response.replace("{" + self.nick+ "}", '')
        response = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>', '', response)
        response = re.sub(r'[()\[\]{}]', '', response)
        response = response.replace("BOT :", '')
        response = response.replace("Bot :", '')
        response = response.replace("bot :", '')
        response = response.replace("BOT :", '')
        response = response.replace("Bot:", '')
        response = response.replace("bot:", '')
        response = response.replace("BOT:", '')
        response = response.replace(":", '')
        responseParams = tokenize(response)

        # Check for commands or recursion or blacklisted words, eg: !generate !generate
        if len(responseParams) > 0 and self.check_if_other_command(responseParams[0]):
            logger.info("You can't make me do commands, you madman!")
            return "You can't make me do commands, you madman!", False
        if self.check_filter(response):
                return "You can't make me say that, you madman!", False
            
        sentenceResponse = " ".join(responseParams.copy())
        sentenceResponse = self.reconstruct_sentence(sentenceResponse)
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

    def send_help_message(self) -> None:
        """Send a Help message to the connected chat, as long as the bot wasn't disabled."""
        if self._enabled:
            logger.info("Help message sent.")
            try:
                self.ws.send_message("Learn how this bot generates sentences here: https://github.com/CubieDev/TwitchMarkovChain#how-it-works")
            except Exception as error:
                logger.warning(f"[{error}] upon sending help message. Ignoring.")
                
    def send_automatic_generation_message(self) -> None:
        global lastSaidMessage
        global nounList
        """Send an automatic generation message to the connected chat.
        
        As long as the bot wasn't disabled, just like if someone typed "!g" in chat.
        """
        try:
            print('!!!!!!!!!!generate time!!!!!!!!')
            print('noun list:')
            print(nounList)
            print(self._enabled)
            cur_time = time.time()
            if self._enabled and self.prev_message_t + self.cooldown < cur_time :
                phraseToUse = self.find_phrase_to_use(nounList)
                print(phraseToUse)
                if phraseToUse is not None:
                    params = tokenize(phraseToUse)
                    sentenceGenerated, success = self.generate(params)
                    if success:
                        # Try to send a message. Just log a warning on fail
                        try:
                            if sentenceGenerated not in '\t'.join(saidMessages):
                                saidMessages.append("{"+ self.nick + "}: " + sentenceGenerated, 360)
                                self.ws.send_message(sentenceGenerated)
                                logger.info("Said Message")
                                self.prev_message_t = time.time()
                            else:
                                logger.info("Tried to say a message, but we saw it was said already")
                            lastSaidMessage = sentenceGenerated
                        except Exception as error:
                            logger.warning(f"[{error}] upon sending help message. Ignoring.")
                    else:
                        logger.info("Attempted to output automatic generation message, but there is not enough learned information yet.")
        except Exception as error:
            logger.warning(f"An error occurred while trying to send an automatic generation message: {error}")
            traceback.print_exc()
            
    def send_whisper(self, user: str, message: str) -> None:
        if self.whisper_cooldown:
            self.ws.send_whisper(user, message)

    def check_filter(self, message: str) -> bool:
        message_lower = message.lower()
        return any(word.lower() in message_lower for word in self.blacklist)

    def check_if_our_command(self, message: str, *commands: "Tuple[str]") -> bool:
        return message.split()[0] in commands

    def check_if_generate(self, message: str) -> bool:
        return self.check_if_our_command(message, *self.generate_commands)
    
    def check_if_other_command(self, message: str) -> bool:
        return message.startswith(("!", "/", ".")) and not message.startswith("/me")
    
    def check_if_permissions(self, m: Message) -> bool:
        return m.user == m.channel or m.user in self.allowed_users

    def check_link(self, message: str) -> bool:
        return self.link_regex.search(message)


if __name__ == "__main__":
    TwitchBot()
