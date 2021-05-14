#ffmpeg -i "jazz.webm" -vn "source-wavs/jazz.wav"
# youtube-dl https://bit.ly/3eYMQjj -f bestaudio
import os   
from pydub import AudioSegment


def save_youtube_audio(link_dict):
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'source-audio')):
        os.mkdir(os.path.join(path_start, 'source-audio'))
    template = "youtube-dl -o '{}/source-audio/{}.%(ext)s' {} -f bestaudio"

    for video, link in link_dict.items():
        cmd = template.format(path_start, video, link)
        print(cmd)
        os.system(cmd)

def convert_to_wav():
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'source-wavs')):
        os.mkdir(os.path.join(path_start, 'source-wavs'))
    template = 'ffmpeg -ss {0} -i "{1}" -vn -to {2} "source-wavs/{3}.wav"'
    audios = os.listdir(os.path.join(path_start, "source-audio"))
    for audio in audios:
        audio_source = "source-audio/{0}".format(audio)
        to = 3600
        start = 0
        if audio == 'lecture.m4a':
            print("done\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            start = 25
        cmd = template.format(start, audio_source, to, audio.split('.')[0])
        os.system(cmd)

def split_tracks(window_size=10000, slide_length=5000):
    path_start = os.getcwd()
    if not os.path.isdir(os.path.join(path_start, 'wav-clips')):
        os.mkdir(os.path.join(path_start, 'wav-clips'))
    tracks = os.listdir(os.path.join(path_start, 'source-wavs'))
    for inc, track in enumerate(tracks):
        song = AudioSegment.from_wav("source-wavs/{0}".format(track))
        increment = 1
        hour = 3600000
        for i in range(0, min(len(song), hour), slide_length):
            print("track: {0}, increment: {1}".format(inc, increment))
            clip = song[i:min(i+window_size, len(song))]
            clip.export("wav-clips/{0}-clip-{1}.wav".format(track.split(".")[0], increment), format="wav")
            increment += 1




if __name__=='__main__':
    link_dict = {'lofi-track-1': 'https://bit.ly/3f5fZcA',
                'lofi-track-2': 'https://bit.ly/3etuXKE',
                'white-noise': 'https://bit.ly/3es6Bkf',
                'city-sounds': 'https://bit.ly/3nWZF1z',
                'jazz': 'https://bit.ly/3bcQyoy',
                'lecture': 'https://bit.ly/3h4t5to'}

    save_youtube_audio(link_dict)
    convert_to_wav()
    split_tracks()
    