"""Train the existing PyTorch ViT-BiLSTM on video-level labels."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from dataset_utils import collect_video_paths, sample_video_frames
from video_model import ViT_BiLSTM

SEED=42
MODEL_DIR=Path('saved_models'); MODEL_DIR.mkdir(exist_ok=True)
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1)
STD=torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1)
class VideoDataset(Dataset):
    def __init__(self, paths, labels, nframes): self.paths,self.labels,self.nframes=paths,labels,nframes
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        arr=sample_video_frames(str(self.paths[i]),frames_per_video=self.nframes,target_size=(224,224))
        x=torch.from_numpy(arr).permute(0,3,1,2).float()
        x=(x-MEAN.squeeze(0))/STD.squeeze(0)
        return x,torch.tensor(int(self.labels[i]),dtype=torch.long)
def evaluate(model, loader, device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); p=torch.softmax(logits,1)[:,1].cpu().numpy()
            ys.extend(y.numpy().tolist()); probs.extend(p.tolist())
    pred=(np.array(probs)>=.5).astype(int); ys=np.array(ys)
    metrics={'accuracy':float(accuracy_score(ys,pred)),'precision':float(precision_score(ys,pred,zero_division=0)), 'recall':float(recall_score(ys,pred,zero_division=0)), 'f1':float(f1_score(ys,pred,zero_division=0)), 'roc_auc':float(roc_auc_score(ys,probs)) if len(set(ys))>1 else 0.0}
    np.savetxt(MODEL_DIR/'confusion_matrix.csv',confusion_matrix(ys,pred),fmt='%d',delimiter=',')
    with open(MODEL_DIR/'metrics_report.json','w') as f: json.dump(metrics,f,indent=2)
    return metrics
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset_dir',default='dataset'); ap.add_argument('--frames_per_video',type=int,default=8); ap.add_argument('--batch_size',type=int,default=1); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--learning_rate',type=float,default=1e-4); args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    paths,labels,_=collect_video_paths(args.dataset_dir)
    trp,tep,trl,tel=train_test_split(paths,labels,test_size=.30,random_state=SEED,stratify=labels)
    vap,tsp,val,tsl=train_test_split(tep,tel,test_size=.50,random_state=SEED,stratify=tel)
    print(f'Videos: {len(paths)} | train {len(trp)} | val {len(vap)} | test {len(tsp)}')
    train=DataLoader(VideoDataset(trp,trl,args.frames_per_video),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(VideoDataset(vap,val,args.frames_per_video),batch_size=1,shuffle=False,num_workers=0)
    test=DataLoader(VideoDataset(tsp,tsl,args.frames_per_video),batch_size=1,shuffle=False,num_workers=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    model=ViT_BiLSTM()
    # Freeze ViT feature extractor first; train temporal layer and classifier.
    for p in model.vit.parameters(): p.requires_grad=False
    model=model.to(device); criterion=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.learning_rate)
    best=-1
    for epoch in range(args.epochs):
        model.train(); total=0; loss_sum=0
        for x,y in train:
            x=x.to(device); y=y.to(device); opt.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); opt.step(); loss_sum+=loss.item(); total+=1
        vm=evaluate(model,val_loader,device); print(f'Epoch {epoch+1}/{args.epochs} train_loss={loss_sum/max(total,1):.4f} val_acc={vm["accuracy"]:.3f} val_f1={vm["f1"]:.3f}',flush=True)
        score=vm['f1']
        if score>best:
            best=score; torch.save({'state_dict':model.state_dict(),'frames_per_video':args.frames_per_video,'image_size':224,'label_map':{'fake':0,'real':1}},MODEL_DIR/'vit_bilstm_video.pth'); print('Saved best model.',flush=True)
    model.load_state_dict(torch.load(MODEL_DIR/'vit_bilstm_video.pth',map_location=device,weights_only=False)['state_dict']); print('TEST:',evaluate(model,test,device))
if __name__=='__main__': main()
