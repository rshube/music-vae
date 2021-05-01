import subprocess
import os
from pydub import AudioSegment



def split_track(track, window_size=10000, slide_length=5000):
    song = AudioSegment.from_mp3("source-mp3s/{}".format(track))
    print(len(song))
    increment = 1
    for i in range(0, len(song), slide_length):
        print(increment)
        clip = song[i:min(i+window_size, len(song))]
        clip.export("clips/{}-clip-{}.mp3".format(track.split(".")[0], increment), format="mp3")
        increment += 1
    
    





if __name__ == "__main__":
    split_track("lofi-track-1.mp3")
    pass