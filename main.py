import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image


class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()

        self.point = 0
        self.new_model = tf.keras.models.load_model("model/save_model.h5")

        self.master = master
        self.master.geometry("1100x550")
        self.master.title("AIアプリ")

        self.create_widgets()

    def create_widgets(self):
        frame0 = tk.Frame(self.master, relief=tk.GROOVE, bd=4)
        frame1 = tk.Frame(frame0)
        label = tk.Label(frame1, text="0~9の数字を書いてね!!", fg="white")
        label.pack()
        frame1.pack(side=tk.TOP, anchor=tk.W)
        frame2 = tk.Frame(frame0, relief=tk.GROOVE, bd=4)
        self.canvas = tk.Canvas(frame2, width=500, height=500, bg="white")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.pack()
        frame2.pack(side=tk.TOP, anchor=tk.W)
        frame0.pack(side=tk.LEFT, anchor=tk.N)

        frame3 = tk.Frame(self.master, relief=tk.GROOVE, bd=4)
        self.button1 = tk.Button(
            frame3, text="CLEAR", command=self.clear, fg="black", bg="blue"
        )
        self.button1.pack(side=tk.LEFT)
        self.button2 = tk.Button(
            frame3,
            text="判定",
            command=self.search_number,
            fg="black",
            bg="blue",
        )
        self.button2.pack(side=tk.LEFT)
        self.button3 = tk.Button(
            frame3, text="終了", command=self.exit_app, fg="black", bg="blue"
        )
        self.button3.pack(side=tk.LEFT)
        frame3.pack(side=tk.TOP, fill="both")

    def plot_percentage(self):
        fig = plt.Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        left = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        height = self.result.flatten()
        ax.barh(left, height, align="center")
        ax.set_xlabel("Percentage [%]")
        ax.set_ylabel("Number")
        ax.set_yticks(left)
        for y, x in zip(left, height):
            ax.text(x, y, "{:.2f}".format(x), ha="left", va="center")
        plt.tight_layout()
        return fig

    def draw(self, event):
        self.canvas.create_oval(
            event.x,
            event.y,
            event.x + 20,
            event.y + 20,
            fill="black",
            outline="black",
        )

    def clear(self):
        if self.point >= 1:
            self.mat_canvas.get_tk_widget().pack_forget()
            self.canvas.delete("all")
            self.point = 0
        else:
            pass

    def search_number(self):
        if self.point == 0:
            self.canvas.postscript(file="img/hogehoge.ps", colormode="color")
            im = Image.open("img/hogehoge.ps")
            im.save("img/hogehoge.jpg")
            img = cv2.imread("img/hogehoge.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_not(img)
            img_resize = cv2.resize(img, dsize=(28, 28))
            img_resize = img_resize / 255.0
            img_resize = np.where(img_resize > 0.3, img_resize, 0)
            img_input = np.ones((1, 28, 28, 1))
            img_input[0, :, :, 0] = img_resize
            self.result = self.new_model.predict(img_input)
            self.fig = self.plot_percentage()
            self.mat_canvas = FigureCanvasTkAgg(self.fig, master=self.master)
            self.mat_canvas.draw()
            self.mat_canvas.get_tk_widget().pack(side=tk.TOP)
            self.point = 1
            plt.close()

    def exit_app(self):
        self.master.destroy()


def main():
    win = tk.Tk()
    app = Application(master=win)
    app.mainloop()


if __name__ == "__main__":
    main()
