# TwitchGPTVector

Twitch Bot for generating messages based on GPT models and VectorDB

---

## Explanation

When the bot starts, it listens to chat messages in the channel listed in the settings.json file. It learns from any chat message that is talking directly to the bot and that is not a question. When someone requests a message to be generated, a GPT model using VectorDB will generate a sentence based on the learned data. Note that the bot is unaware of the meaning of any of its inputs and outputs. This means it can use bad language if it was taught to use bad language by people in chat. You can add a list of banned words it should never learn or say. Use at your own risk.
---

## How it works

### Sentence Parsing

The bot processes chat messages, extracts their keywords and nouns, using those as the subject for it's prompt. It creates a semantic vector from the message and stores it in the vectordb, so that previous messages about that subject can be retrieved.

---

### Generation

When a message is generated with !generate, the GPT model and VectorDB work together to create a message based on the learned data. The generated message will resemble the structure and content of the chat messages the bot has learned from.  It will use context from the last few minutes of chat, as well as some context from its long term database.

## Commands

Chat members can generate chat-like messages using the following commands (Note that they are aliases):

```txt
!generate [words]
!g [words]
```

Example:

```txt
!g Curly
```

Result (for example):

```txt
Curly fries are the reason I don't go to the movies anymore
```

---

### Streamer commands

All of these commands can be whispered to the bot account, or typed in chat.
To disable the bot from generating messages, while still learning from regular chat messages:

```txt
!disable
```

After disabling the bot, it can be re-enabled using:

```txt
!enable
```

Changing the cooldown between generations is possible with one of the following two commands:

```txt
!setcooldown <seconds>
!setcd <seconds>
```

Example:

```txt
!setcd 30
```

Which sets the cooldown between generations to 30 seconds.

---

### Moderator commands

All of these commands must be whispered to the bot account.
Moderators (and the broadcaster) can modify the blacklist to prevent the bot learning words it shouldn't.
To add `word` to the blacklist, a moderator can whisper the bot:

```txt
!blacklist <word>
```

Similarly, to remove `word` from the blacklist, a moderator can whisper the bot:

```txt
!whitelist <word>
```

And to check whether `word` is already on the blacklist or not, a moderator can whisper the bot:

```txt
!check <word>
```

---

## Settings

This bot is controlled by a `settings.json` file, which has the following structure:

```json
{
  "Host": "irc.chat.twitch.tv",
  "Port": 6667,
  "Channel": "#<channel>",
  "Nickname": "<name>",
  "Authentication": "oauth:<twitchauth>",
  "OpenAIKey": "<openAIoauth>",
  "DeniedUsers": ["StreamElements", "Nightbot", "Moobot", "Marbiebot"],
  "AllowedUsers": [],
  "Cooldown": 20,
  "HelpMessageTimer": 18000,
  "AutomaticGenerationTimer": -1,
  "WhisperCooldown": true,
  "EnableGenerateCommand": true,
  "AllowGenerateParams": true,
  "GenerateCommands": ["!generate", "!g"]
}
```

| **Parameter**              | **Meaning**                                                                                                                                                                                                                                  | **Example**                                             |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `Host`                     | The URL that will be used. Do not change.                                                                                                                                                                                                    | `"irc.chat.twitch.tv"`                                  |
| `Port`                     | The Port that will be used. Do not change.                                                                                                                                                                                                   | `6667`                                                  |
| `Channel`                  | The Channel that will be connected to.                                                                                                                                                                                                       | `"#Starstorm"`                                           |
| `Nickname`                 | The Username of the bot account.                                                                                                                                                                                                             | `"Starstorm_v2"`                                            |
| `Authentication`           | The OAuth token for the bot account.                                                                                                                  | `"oauth:pivogip8ybletucqdz4pkhag6itbax"`                |
| `ClientID`                 | The authentication key for pulling emotes, and other things we can't do without registering with twitch. You will need to go to https://dev.twitch.tv/console/apps/create and create a application.                                                                                                               | `"oauth:pivogip8ybletucqdz4pkhag6itbax"`                |                                                                                                                                                                                             | `"oauth:pivogip8ybletucqdz4pkhag6itbax"`                |
| `OpenAIAuth`               | The authentication key for open ai. You will need to go to https://platform.openai.com/account/api-keys and create an account.                                                                                                                                                                                                         | `"asdfasdfasdf"`                |
| `DeniedUsers`              | The list of (bot) accounts whose messages should not be learned from. The bot itself it automatically added to this.                                                                                                                         | `["StreamElements", "Nightbot", "Moobot", "Marbiebot"]` |
| `AllowedUsers`             | A list of users with heightened permissions. Gives these users the same power as the channel owner, allowing them to bypass cooldowns, set cooldowns, disable or enable the bot, etc.                                                        | `["loltyler1", "starstorm"]`                                 |
| `Cooldown`                 | A cooldown in seconds between successful generations. If a generation fails (eg inputs it can't work with), then the cooldown is not reset and another generation can be done immediately.                                                   | `20|
| `AutomaticGenerationTimer` | The amount of seconds between automatically sending a generated message, as if someone wrote `!g`. -1 for no automatic generations.                                                                                                          | `-1`                                                    |
| `WhisperCooldown`          | Allows the bot to whisper a user the remaining cooldown after that user has attempted to generate a message.                                                                                                                                 | `true`                                                  |
| `EnableGenerateCommand`    | Globally enables/disables the generate command.                                                                                                                                                                                              | `true
| `GenerateCommands`         | The generation commands that the bot will listen for. Defaults to `["!generate", "!g"]`. Useful if your chat is used to commands with `~`, `-`, `/`, etc.                                                                                    | `["!generate", "!g"]`                                   |

_Note that the example OAuth token is not an actual token, but merely a generated string to give an indication what it might look like._

I got my real OAuth token from <https://twitchapps.com/tmi/>.

---

### Blacklist

You may add words to a blacklist by adding them on a separate line in `blacklist.txt`. Each word is case insensitive. By default, this file only contains `<start>` and `<end>`, which are required for the current implementation.

Words can also be added or removed from the blacklist via whispers, as is described in the [Moderator Command](#moderator-commands) section.

---

## Requirements

- [Python 3.6+](https://www.python.org/downloads/)
- [Module requirements](requirements.txt)
  - Install these modules using `pip install -r requirements.txt` in the commandline.


Additionally, you'll need to have the GPT model files and access to the Vector DB for the bot to function correctly.
whichever model you want from https://nlp.stanford.edu/projects/glove/
spacy en_core_web_sm
https://github.com/spotify/annoy
---

Credits to CubieDev, this was originally his github project i repurposed: https://github.com/tomaarsen/TwitchMarkovChain
