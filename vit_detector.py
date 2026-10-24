"""PyTorch inference adapter used by the existing Flask app."""
import logging
from pathlib import Path
import cv2, numpy as np, torch
from video_model import ViT_BiLSTM
logger=logging.getLogger(__name__)
class DeepfakeDetector:
    def __init__(self):
        self.model=None; self.frames_per_video=8; self.image_size=224; self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.model_loaded=False; self.load_model()
    def load_model(self):
        path=Path(__file__).resolve().parent/'saved_models'/'vit_bilstm_video.pth'
        if not path.exists(): logger.warning('Trained PyTorch model not found at %s',path); self.model_loaded=False; return False
        try:
            checkpoint=torch.load(path,map_location=self.device,weights_only=False); self.frames_per_video=int(checkpoint.get('frames_per_video',8)); self.model=ViT_BiLSTM(); self.model.load_state_dict(checkpoint['state_dict']); self.model.to(self.device).eval(); self.model_loaded=True; return True
        except Exception: logger.exception('Unable to load PyTorch video model'); self.model=None; self.model_loaded=False; return False
    def predict_video(self,video_path):
        if not self.model_loaded: return None,'Trained PyTorch model not loaded. Train the model first.'
        cap=cv2.VideoCapture(str(video_path))
        try:
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 30); 
            if total<=0 or not cap.isOpened(): return None,'Could not open video or no frames found.'
            indices=np.linspace(0,total-1,self.frames_per_video,dtype=int); frames=[]; i=0; target=0
            while target<len(indices):
                ok,frame=cap.read()
                if not ok: break
                if i==indices[target]:
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); frame=cv2.resize(frame,(224,224)); frames.append(frame.astype(np.float32)/255.); target+=1
                i+=1
            if not frames:return None,'Could not extract frames.'
            while len(frames)<self.frames_per_video: frames.append(frames[-1].copy())
            x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float().unsqueeze(0)
            mean=torch.tensor([.485,.456,.406]).view(1,1,3,1,1); std=torch.tensor([.229,.224,.225]).view(1,1,3,1,1); x=(x-mean)/std
            with torch.no_grad(): p=float(torch.softmax(self.model(x.to(self.device)),dim=1)[0,1].item())
            fake=1-p; label='REAL' if p>=.5 else 'FAKE'
            return {'prediction':label,'confidence':max(p,fake),'real_probability':p,'fake_probability':fake,'frames_analyzed':len(frames),'total_frames':total,'fps':fps,'duration':total/max(fps,1),'selected_frames':len(frames)},None
        except Exception as e: logger.exception('Prediction failed'); return None,str(e)
        finally: cap.release()
