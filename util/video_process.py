from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
class VideoLoader:
    def __init__(self, videoUrl):
        self.videoUrl = videoUrl
        self.clip = VideoFileClip(videoUrl)
        self.cap = cv2.VideoCapture(videoUrl)
        
    def get_video_info(self):
        fps = self.clip.fps
        w, h = self.clip.size
        video_duration_frames = int(self.clip.duration * fps)
        return fps, w, h, video_duration_frames
    
    def edit_video(self, intervals, output_path = 'temp.mp4'):
        clips = []
        for interval in intervals:
            start, end = interval
            clip = self.clip.subclip(start, end)
            clips.append(clip)
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path)
    
    # 读取视频帧
    def read_frame(self):
        return self.cap.read()