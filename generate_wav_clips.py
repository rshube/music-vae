import os 
from utils import DATA_PATH  
from pydub import AudioSegment


def save_youtube_audio(link_dict):
    save_dir = os.path.join(DATA_PATH, 'source-audio')
    os.makedirs(save_dir, exist_ok=True)

    template = "youtube-dl -o '{}/{}.%(ext)s' {} -f bestaudio"
    for video, link in link_dict.items():
        cmd = template.format(save_dir, video, link)
        print(cmd)
        os.system(cmd)

def convert_to_wav():
    save_dir = os.path.join(DATA_PATH, 'source-wavs')
    os.makedirs(save_dir, exist_ok=True)
    source_dir = os.path.join(DATA_PATH, 'source-audio')

    template = 'ffmpeg -ss {0} -i "{1}" -vn -to {2} "{3}"'
    audios = os.listdir(source_dir)
    for audio in audios:
        audio_source = f'{source_dir}/{audio}'
        wav_dest = f'{save_dir}/{audio.split(".")[0]}.wav'
        to = 3600
        start = 0
        if audio == 'lecture.m4a':
            start = 25
        cmd = template.format(start, audio_source, to, wav_dest)
        os.system(cmd)

def split_tracks(window_size=10000, slide_length=5000):
    save_dir = os.path.join(DATA_PATH, 'wav-clips')
    os.makedirs(save_dir, exist_ok=True)
    source_dir = os.path.join(DATA_PATH, 'source-wavs')

    tracks = os.listdir(source_dir)
    for track in tracks:
        song = AudioSegment.from_wav(f'{source_dir}/{track}')
        increment = 1
        hour = 3600000
        track_name = track.split(".")[0]
        for i in range(0, min(len(song), hour), slide_length):
            print(f"track: {track_name}, increment: {increment}")
            clip = song[i:min(i+window_size, len(song))]
            clip.export(f'{save_dir}/{track_name}-clip-{increment}.wav', format="wav")
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
    