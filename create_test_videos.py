import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path, num_frames=300, width=224, height=224):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print(f"Created: {output_path}")

Path("dataset/real").mkdir(exist_ok=True, parents=True)
Path("dataset/fake").mkdir(exist_ok=True, parents=True)

for i in range(5):
    create_test_video(f"dataset/real/real_test_{i}.mp4")
    
for i in range(5):
    create_test_video(f"dataset/fake/fake_test_{i}.mp4")
    
print("Test videos created!")
