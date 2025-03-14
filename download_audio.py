import yt_dlp

def download_audio(youtube_url, output_folder="downloads"):
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Change to 'wav' or 'm4a' if needed
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([youtube_url])

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    download_audio(url)