# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 00:51
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import shutil
import threading


class VideoKeyFrameEventAnnotator(object):
    """
    视频关键帧事件标注工具
    """

    def __init__(self, main_window):
        # 初始化一重要参数
        self.cap = None
        self.video_path = None
        self.frame_count = -1
        self.video_frames_save_root_dir = None
        self.video_frames_save_dir = None
        self.current_frame_id = -1
        self.play_interval = 1000 / 30
        self.frame_width = 1920
        self.frame_height = 1080
        self.width_diff = 176
        self.height_diff = 24

        # 初始化自动播放的线程
        self.is_playing_evnet = threading.Event()
        self.is_playing_evnet.clear()
        self.auto_play_thread = threading.Thread(target=self._auto_play, args=(self.is_playing_evnet,), daemon=True)
        self.auto_play_thread.start()

        # 初始化主窗口
        self.main_window = main_window
        self.main_window.title("视频关键帧事件标注工具")

        # 视频帧图像显示区域
        self.display_label = tk.Label(self.main_window)
        self.display_label.grid(row=0, column=0, padx=10, pady=10)
        self.show_frame(frame_idx=None)  # 打开时显示空白图像

        # 功能按钮区域
        self.button_frame = tk.Frame(self.main_window)
        self.button_frame.grid(row=0, column=1, sticky='n')

        # 依次添加功能按钮
        # 打开视频按钮
        self.open_video_button = tk.Button(self.button_frame, text="打开视频", command=self.load_video).grid(row=0, column=0, padx=5, pady=10, sticky='n')
        # 开始解析按钮
        self.start_extract_button = tk.Button(self.button_frame, text="解析视频", command=self.extract_video).grid(row=0, column=1, padx=5, pady=10, sticky='n')
        # 上一帧按钮
        self.pre_frame_button = tk.Button(self.button_frame, text="上一帧", command=lambda: self.change_displayed_frame(-1)).grid(row=1, column=0, padx=5, pady=10, sticky='n')
        self.main_window.bind("<Left>", lambda event: self.change_displayed_frame(-1))
        # 下一帧按钮
        self.next_frame_button = tk.Button(self.button_frame, text="下一帧", command=lambda: self.change_displayed_frame(1)).grid(row=1, column=1, padx=5, pady=10, sticky='n')
        self.main_window.bind("<Right>", lambda event: self.change_displayed_frame(1))
        # 1倍速播放按钮
        self.x1_speed_play_button = tk.Button(self.button_frame, text="1倍速播放", command=lambda: self._x_speed_play(1)).grid(row=2, column=0, padx=5, pady=10, sticky='n')
        # 2倍速播放按钮
        self.x2_speed_play_button = tk.Button(self.button_frame, text="2倍速播放", command=lambda: self._x_speed_play(2)).grid(row=2, column=1, padx=5, pady=10, sticky='n')
        # 3倍速播放按钮
        self.x3_speed_play_button = tk.Button(self.button_frame, text="3倍速播放", command=lambda: self._x_speed_play(3)).grid(row=3, column=0, padx=5, pady=10, sticky='n')
        # 暂停播放按钮
        self.pause_play_button = tk.Button(self.button_frame, text="暂停播放", command=self.pause_play).grid(row=3, column=1, padx=5, pady=10, sticky='n')

        # 创建一个水平分界线
        separator = tk.Frame(self.button_frame, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=4, column=0, columnspan=2)

    def show_frame(self, frame_idx=None):
        if frame_idx is None:
            # 创建一个空白的透明图像作为label的大小占位符
            empty_image = Image.new('RGB', (1920, 1080), (255, 255, 255))  # 创建一个800x600的白色图像
            img = ImageTk.PhotoImage(empty_image)
            self.display_label.config(image=img)
            self.display_label.image = img
        else:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(frame)
                    img = ImageTk.PhotoImage(image=im)
                    self.display_label.config(image=img)
                    self.display_label.image = img

    def _create_progress_window_bar(self):
        # 创建顶层窗口
        progress_window = tk.Toplevel(self.main_window)
        progress_window.title("视频解析进度")
        progress_window.geometry("300x100+{}+{}".format(self.main_window.winfo_x() + self.main_window.winfo_width() // 2 - 150, self.main_window.winfo_y() + self.main_window.winfo_height() // 2 - 50))
        # 创建关闭进度条窗口的事件
        close_progress_window_event = threading.Event()
        # 绑定关闭进度条窗口事件
        progress_window.protocol("WM_DELETE_WINDOW", lambda: close_progress_window_event.set())
        # 创建进度条
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=280, mode="determinate")
        progress_bar['maximum'] = 100  # 进度值最大值
        progress_bar['value'] = 0  # 进度值初始值
        progress_bar.pack(pady=20, padx=10)
        progress_bar.update_idletasks()

        return progress_bar, progress_window, close_progress_window_event

    def _extract_video(self, progress_bar, progress_window, close_progress_window_event):
        # 打开临时视频流
        temp_cap = cv2.VideoCapture(self.video_path)
        count = 0
        while True:
            ret, frame = temp_cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(self.video_frames_save_dir, f'frame_{count:08d}.jpg'), frame)
            count += 1
            if int(100 * count / self.frame_count) > int(100 * (count - 1) / self.frame_count):
                progress_bar['value'] = int(100 * count / self.frame_count)  # 更新进度条的当前进度
                progress_window.update_idletasks()  # 更新GUI
            # 判断是否结束线程
            if close_progress_window_event.is_set():
                # 关闭临时视频流
                temp_cap.release()
                # 销毁顶层窗口和进度条
                progress_window.destroy()
        # 关闭临时视频流
        temp_cap.release()
        # 销毁顶层窗口和进度条
        progress_window.destroy()

    def extract_video(self):
        self.is_playing_evnet.clear()
        if self.video_frames_save_root_dir:
            self.video_frames_save_dir = os.path.join(self.video_frames_save_root_dir, os.path.basename(self.video_path).split('.')[0])
            # 判断是否存在当前目录且目录下的图像数等于视频帧数，满足则不用重新解析了
            if os.path.exists(self.video_frames_save_dir) and len(os.listdir(self.video_frames_save_dir)) == self.frame_count:
                return
            # 如果没有解析过或者解析不完全，则执行视频解析
            if os.path.exists(self.video_frames_save_dir):
                shutil.rmtree(self.video_frames_save_dir)
            os.makedirs(self.video_frames_save_dir, exist_ok=False)  # 创建视频所有帧图像的存储目录
            # 创建顶层窗口和进度条
            progress_bar, progress_window, close_progress_window_event = self._create_progress_window_bar()
            # 启动一个线程解析视频
            extract_video_thread = threading.Thread(target=self._extract_video, args=(progress_bar, progress_window, close_progress_window_event), daemon=True)
            extract_video_thread.start()

    def update_main_window_position(self):
        self.main_window_width = self.frame_width + self.width_diff
        self.main_window_height = self.frame_height + self.height_diff
        self.main_window_offset_x = int((self.main_window.winfo_screenwidth() - self.main_window_width) / 2)
        self.main_window_offset_y = int((self.main_window.winfo_screenheight() - self.main_window_height) / 2)
        self.main_window.geometry(f"{self.main_window_width}x{self.main_window_height}+{self.main_window_offset_x}+{self.main_window_offset_y}")

    def load_video(self):
        self.is_playing_evnet.clear()
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            # 打开视频流
            self.cap = cv2.VideoCapture(self.video_path)
            # 获取视频帧的宽度和高度
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 默认显示第一帧
            self.current_frame_id = 0
            self.show_frame(frame_idx=self.current_frame_id)
            # 更新主窗口位置
            self.update_main_window_position()
            # 获取视频总帧数
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 选择解析的视频帧存放目录
            self.video_frames_save_root_dir = filedialog.askdirectory()

    def change_displayed_frame(self, step):
        self.is_playing_evnet.clear()
        self._change_displayed_frame(step)

    def _change_displayed_frame(self, step):
        if self.cap:
            self.current_frame_id += step
            self.current_frame_id = max(0, min(self.frame_count - 1, self.current_frame_id))
            self.show_frame(frame_idx=self.current_frame_id)

    def _auto_play(self, is_playing_evnet):
        while True:
            # 事件等待，只有事件被设置时，才会继续执行
            is_playing_evnet.wait()
            # 一段时间后执行跳转下一帧
            time.sleep(self.play_interval / 1000)
            self._change_displayed_frame(1)

    def _x_speed_play(self, speed):
        self.play_interval = 1000 / 30 / speed
        self.is_playing_evnet.set()

    def pause_play(self):
        self.is_playing_evnet.clear()


class VideoFrameExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("视频关键帧标注工具")

        # 视频帧显示区域
        self.display_label = tk.Label(self.root)
        self.display_label.grid(row=0, column=0, padx=10, pady=10)
        self.show_frame(frame_idx=None)

        # 功能按钮区域
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=1, sticky='n')

        # 打开视频文件按钮
        tk.Button(self.button_frame, text="打开视频", command=self.load_video).pack()

        # 解析视频按钮
        tk.Button(self.button_frame, text="解析视频", command=self.extract_frames).pack()

        # 上一帧、下一帧按钮
        tk.Button(self.button_frame, text="上一帧", command=lambda: self.change_frame(-1)).pack()
        tk.Button(self.button_frame, text="下一帧", command=lambda: self.change_frame(1)).pack()

        # 自动播放视频按钮
        self.play_speed = 1000 // 30  # 按30FPS播放
        self.play_button = tk.Button(self.button_frame, text="播放视频", command=self.auto_play)
        self.play_button.pack()

        # 加载视频及初始化变量
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.playing = False

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.extract_frames(file_path)
            self.show_frame(0)

    def extract_frames(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename()
        if file_path:
            cap = cv2.VideoCapture(file_path)
            count = 0
            save_path = filedialog.askdirectory()
            if save_path:
                save_path = os.path.join(save_path, os.path.basename(file_path).split('.')[0])
                os.makedirs(save_path, exist_ok=True)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(os.path.join(save_path, f'frame_{count:04d}.jpg'), frame)
                    print(os.path.join(save_path, f'frame_{count:04d}.jpg'))
                    count += 1
            cap.release()

    def change_frame(self, step):
        if self.cap:
            self.current_frame += step
            self.current_frame = max(0, min(self.frame_count - 1, self.current_frame))
            self.show_frame(self.current_frame)

    def show_frame(self, frame_idx=None):
        if frame_idx is None:
            # 创建一个空白的透明图像作为label的大小占位符
            empty_image = Image.new('RGB', (1920, 1080), (255, 255, 255))  # 创建一个800x600的白色图像
            img = ImageTk.PhotoImage(empty_image)
            self.display_label.config(image=img)
            self.display_label.image = img
            return
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                self.display_label.config(image=img)
                self.display_label.image = img

    def auto_play(self):
        if not self.playing:
            self.playing = True
            self.play_button.config(text="暂停播放")
            self.play_next_frame()
        else:
            self.playing = False
            self.play_button.config(text="播放视频")

    def play_next_frame(self):
        if self.playing and self.cap:
            self.change_frame(1)
            self.root.after(self.play_speed, self.play_next_frame)


if __name__ == '__main__':
    # 创建主窗口
    main_window = tk.Tk()
    app = VideoKeyFrameEventAnnotator(main_window)
    # 更新主窗口位置
    app.update_main_window_position()
    main_window.mainloop()
