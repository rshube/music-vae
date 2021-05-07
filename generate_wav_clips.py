#ffmpeg -i "jazz.webm" -vn "source-wavs/jazz.wav"
# youtube-dl https://bit.ly/3eYMQjj -f bestaudio
import os   
from pydub import AudioSegment


def save_youtube_audio(link_dict, fake_link):
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'source-audio')):
        os.mkdir(os.path.join(path_start, 'source-audio'))
    template = "youtube-dl -o '{}/source-audio/{}.%(ext)s' {} -f bestaudio"

    for video in link_dict:
        link = link_dict[video]
        #link = fake_link
        cmd = template.format(path_start, video, link)
        os.system(cmd)

def convert_to_wav():
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'source-wavs')):
        os.mkdir(os.path.join(path_start, 'source-wavs'))
    template = 'ffmpeg -ss {} -to {} -i "{}" -vn "source-wavs/{}.wav"'
    audios = os.listdir(os.path.join(path_start, "source-audio"))
    for audio in audios:
        audio_source = "source-audio/{}".format(audio)
        to = 3600
        start = 0
        if audio == 'lecture.m4a':
            print("done\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            start = 25
        cmd = template.format(start, to, audio_source, audio.split('.')[0])
        os.system(cmd)

def split_tracks(window_size=10000, slide_length=5000):
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'wav-clips')):
        os.mkdir(os.path.join(path_start, 'wav-clips'))
    tracks = os.listdir(os.path.join(path_start, 'source-wavs'))
    for inc, track in enumerate(tracks):
        song = AudioSegment.from_wav("source-wavs/{}".format(track))
        increment = 1
        hour = 3600000
        for i in range(0, min(len(song), hour), slide_length):
            print("track: {}, increment: {}".format(inc, increment))
            clip = song[i:min(i+window_size, len(song))]
            clip.export("wav-clips/{}-clip-{}.wav".format(track.split(".")[0], increment), format="wav")
            increment += 1




if __name__=='__main__':
    link_dict = {'lofi-track-1': 'https://bit.ly/3etuXKE', 
                'lofi-track-2': 'https://bit.ly/3f5fZcA',
                'white-noise': 'https://bit.ly/3es6Bkf',
                'city-sounds': 'https://bit.ly/3nWZF1z',
                'jazz': 'https://bit.ly/3bcQyoy',
                'lecture': 'https://bit.ly/3h4t5to'}

    fake_link = 'https://bit.ly/3uv9TJ1'

    save_youtube_audio(link_dict, fake_link)
    convert_to_wav()
    split_tracks()
    