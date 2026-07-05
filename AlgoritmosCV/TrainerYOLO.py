from ultralytics import YOLO
import os

def train():

    model = YOLO('yolov8n.pt')

    model.train(
        data=r'C:\Users\Gabriel\TCC_GEMARS\data.yaml',
        epochs=300,
        imgsz=640,
        rect=False,
        mosaic=1.0,
        augment=True,      
        patience=100,      
        optimizer='SGD',
        project='TCC_GEMARS',
        name='TCCGEMARSYOLO8ATT',
        device=0,          
        batch=8,           
        workers=4,
        plots=True,        
        save=True         
    )

if __name__ == '__main__':
    train()