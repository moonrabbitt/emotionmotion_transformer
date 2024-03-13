# Prototype 1

## Week 1

### Initial Development and Testing

This week, I focused on developing the first prototype, centered around creating a functional broadcast loop. The process involves scraping live chat data from a YouTube stream, converting this data into a sentiment score, and then using this sentiment to influence the motion of the artwork. The motion is then broadcast back on the same YouTube live stream, forming a closed loop.

My initial approach was to scrape YouTube comments using a Selenium script. However, I discovered that YouTube comments don't update in real time and have about an hour's delay. Therefore, I shifted my attention to live chat data. Although I initially considered using the YouTube API for scraping live chat, this proved to be complicated and limited in terms of the number of live chats I could scrape. I then found and utilized the [pytchat](https://github.com/taizan-hokuto/pytchat) library, which efficiently handled the live chat scraping.

As a proof of concept, I recorded short sequences of myself performing simple movements. I assigned these videos numbers from 1 to 5 and then entered these numbers into the live chat as commands to select which video should be played. The pytchat library scrapes the live chat and selects the corresponding video. For broadcasting the video back into the YouTube live stream, I used [OBS Studio](https://obsproject.com), thereby creating a closed loop.

The results of the first prototype can be seen in the video below:

[Prototype 1](https://drive.google.com/file/d/1-d9H-FqGFaE9BaBBq-JiywU8QJSF3O9o/view?usp=drive_link)



## Week 2

### Improving the Broadcast Loop

The initial setup of the broadcast loop involved manual intervention, which included starting the stream on OBS, and manually copying and pasting the YouTube live stream's HTML into my script. This manual process proved to be cumbersome, especially when it came to debugging, as it had to be repeated each time.

To enhance efficiency, I attempted to automate the entire system using the OBS web driver. However, this approach encountered significant challenges, primarily due to OBS's persistent requirement for manual confirmations. Despite these difficulties, I managed to refine the process to some extent. By incorporating parsing functions into the script, I streamlined the setup procedure, reducing the likelihood of errors and improving ease of use.

The updated code, reflecting these modifications, is available in the following [notebook](notebooks/prototypes/basic-prototype1.ipynb).
