import tkinter as tk
from tkinter import ttk, messagebox, Canvas, filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import os

#分类名称
class_names = [
    'T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
    '凉鞋', '衬衫', '运动鞋', '包', '短靴'
]

#加载数据和模型
print("正在加载模型...")
(_, _), (test_images, test_labels) = datasets.fashion_mnist.load_data()
test_images = test_images / 255.0
model = tf.keras.models.load_model("fashion_mnist_model.h5")

#创建主窗口
root = tk.Tk()
root.title("Fashion MNIST 服装分类器")
root.geometry("520x700")

#全局变量，用于保存显示的原图（防止被回收）
current_display_img = None

#标题
title_label = ttk.Label(root, text="服装图片分类系统", font=("微软雅黑", 18))
title_label.pack(pady=10)

#绘图区域
canvas = Canvas(root, width=280, height=280, bg="white")
canvas.pack(pady=10)

#结果标签
result_label = ttk.Label(
    root,
    text="输入序号测试数据集 / 选择本地图片识别",
    font=("微软雅黑", 14)
)
result_label.pack(pady=10)

#手动输入序号区域
input_frame = ttk.Frame(root)
input_frame.pack(pady=5)
ttk.Label(input_frame, text="图片序号（0-9999）：", font=("微软雅黑", 12)).grid(row=0, column=0)
index_entry = ttk.Entry(input_frame, font=("微软雅黑", 12), width=10)
index_entry.grid(row=0, column=1, padx=5)
index_entry.focus()

#绘制测试集的28x28图片
def draw_image(img):
    canvas.delete("all")
    scale = 10
    for i in range(28):
        for j in range(28):
            gray = img[i][j]
            color = f'#{int(255 * (1 - gray)):02x}{int(255 * (1 - gray)):02x}{int(255 * (1 - gray)):02x}'
            canvas.create_rectangle(
                j * scale, i * scale,
                (j + 1) * scale, (i + 1) * scale,
                fill=color, outline=color
            )

#显示本地原图（缩放到画布大小）
def show_original_image(image_path):
    global current_display_img
    try:
        img = tk.PhotoImage(file=image_path)
        scale = min(280 / img.width(), 280 / img.height())
        # 计算缩放比例
        subsample_factor = int(1 / scale) if scale < 1 else 1
        img = img.subsample(subsample_factor)
        current_display_img = img
        canvas.delete("all")
        canvas.create_image(140, 140, image=current_display_img)
    except Exception as e:
        messagebox.showerror("图片显示失败", f"{str(e)}")

#根据序号预测测试集图片
def predict_by_index(idx):
    if idx < 0 or idx >= len(test_images):
        messagebox.showerror("错误", "序号必须在 0~9999 之间！")
        return
    img = test_images[idx]
    true_label = test_labels[idx]
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_idx = np.argmax(pred)
    confidence = np.max(tf.nn.softmax(pred)) * 100
    draw_image(img)
    result_label.config(text=f"""
图片序号：{idx}
真实类别：{class_names[true_label]}
预测类别：{class_names[pred_idx]}
置信度：{confidence:.2f}%
""")

#用TensorFlow处理图片
def load_and_process_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=1)
        img = tf.image.resize(img, (28, 28))
        img = img / 255.0
        img = 1.0 - img
        img_array = np.squeeze(img.numpy())
        return img_array
    except Exception as e:
        raise Exception(f"图片处理失败：{str(e)}")

#选择本地图片并预测
def predict_local_image():
    file_path = filedialog.askopenfilename(
        title="选择本地图片",
        filetypes=[("图片文件", "*.png *.jpg *.jpeg"), ("所有文件", "*.*")]
    )
    if not file_path:
        return
    try:
        img_arr = load_and_process_image(file_path)
        pred = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
        pred_idx = np.argmax(pred)
        confidence = np.max(tf.nn.softmax(pred)) * 100
        # 显示原图
        show_original_image(file_path)
        result_label.config(text=f"""
本地图片：{os.path.basename(file_path)}
预测类别：{class_names[pred_idx]}
置信度：{confidence:.2f}%
（本地图片无真实标签）
""")
    except Exception as e:
        messagebox.showerror("识别失败", f"{str(e)}")

#手动查询
def query_manual():
    try:
        idx = int(index_entry.get())
        predict_by_index(idx)
    except ValueError:
        messagebox.showerror("错误", "请输入有效数字！")

#随机测试
def query_random():
    idx = np.random.randint(0, len(test_images))
    index_entry.delete(0, tk.END)
    index_entry.insert(0, str(idx))
    predict_by_index(idx)

#按钮布局
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=15)
ttk.Button(btn_frame, text="手动查询该序号", command=query_manual, width=15).grid(row=0, column=0, padx=5)
ttk.Button(btn_frame, text="随机测试一张", command=query_random, width=15).grid(row=0, column=1, padx=5)

ttk.Button(root, text="选择本地图片识别", command=predict_local_image, width=20).pack(pady=5)
ttk.Button(root, text="退出程序", command=root.quit).pack(pady=5)

#启动提示
if __name__ == "__main__":
    root.after(100, lambda: messagebox.showinfo("启动成功",
        "模型加载完成！\n序号：0 ~ 9999\n支持格式：PNG / JPG / JPEG"))
    root.mainloop()